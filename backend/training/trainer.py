"""
Training engine for DeepScan deepfake detection model.

Features:
- Mixed-precision training (AMP) for CUDA and MPS
- Layer-wise learning rate decay (backbone gets 10x lower LR than head)
- Cosine annealing with linear warmup
- Gradient clipping
- Early stopping with patience
- Best-checkpoint saving
- Per-epoch metrics (accuracy, F1, AUC-ROC)
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from sklearn.metrics import f1_score, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── LR group builder ─────────────────────────────────────────────────────────

def _build_param_groups(model: nn.Module, base_lr: float, backbone_lr_ratio: float = 0.1):
    """
    Split parameters into backbone (lower LR) and head (base LR) groups.
    Frozen parameters (requires_grad=False) are excluded automatically.
    """
    backbone_params, head_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # backbone / dino / clip / vit are pretrained — lower LR
        if any(k in name for k in ('backbone', 'dino', 'clip', 'vit', 'efficientnet')):
            backbone_params.append(param)
        else:
            head_params.append(param)

    return [
        {'params': backbone_params, 'lr': base_lr * backbone_lr_ratio},
        {'params': head_params,     'lr': base_lr},
    ]


# ── Warmup scheduler ─────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup then cosine decay, wrapping a CosineAnnealingLR."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self._cosine = CosineAnnealingLR(
            optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=1e-7
        )
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch <= self.warmup_epochs:
            scale = self.last_epoch / max(self.warmup_epochs, 1)
            for g, base in zip(self.optimizer.param_groups, self.base_lrs):
                g['lr'] = base * scale
        else:
            self._cosine.step()

    def get_last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _accuracy(preds: list[int], labels: list[int]) -> float:
    return sum(p == l for p, l in zip(preds, labels)) / len(labels)


def _compute_metrics(all_labels, all_preds, all_probs):
    acc = _accuracy(all_preds, all_labels)
    metrics = {'accuracy': round(acc * 100, 2)}
    if HAS_SKLEARN:
        metrics['f1'] = round(f1_score(all_labels, all_preds, average='binary', zero_division=0), 4)
        try:
            metrics['auc'] = round(roc_auc_score(all_labels, all_probs), 4)
        except Exception:
            metrics['auc'] = 0.0
    return metrics


# ── One-epoch pass ────────────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, device, scaler, is_train: bool):
    model.train() if is_train else model.eval()

    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type,
                                dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16,
                                enabled=device.type in ('cuda', 'mps')):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits.float(), dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    metrics = _compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics


# ── Main Trainer ──────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Path,
        device: torch.device,
        epochs: int = 20,
        base_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 2,
        early_stopping_patience: int = 5,
        backbone_lr_ratio: float = 0.1,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.epochs = epochs
        self.patience = early_stopping_patience

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Label smoothing helps generalisation; sampler already handles class balance
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        param_groups = _build_param_groups(model, base_lr, backbone_lr_ratio)
        self.optimizer = AdamW(param_groups, weight_decay=weight_decay)
        self.scheduler = WarmupCosineScheduler(self.optimizer, warmup_epochs, epochs)

        # AMP scaler — CUDA only (MPS uses bfloat16 natively, no scaler needed)
        self.scaler = torch.cuda.GradScaler() if device.type == 'cuda' else None

        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history: list[dict] = []

    def _log(self, epoch, train_loss, train_m, val_loss, val_m, elapsed):
        lr = self.scheduler.get_last_lr()[-1]
        line = (
            f"Epoch {epoch:03d}/{self.epochs} | "
            f"LR {lr:.2e} | "
            f"Train loss {train_loss:.4f} acc {train_m['accuracy']:.1f}% | "
            f"Val loss {val_loss:.4f} acc {val_m['accuracy']:.1f}%"
        )
        if 'f1' in val_m:
            line += f" f1 {val_m['f1']:.3f}"
        if 'auc' in val_m:
            line += f" auc {val_m['auc']:.3f}"
        line += f" | {elapsed:.0f}s"
        print(line)

    def _save_checkpoint(self, epoch: int, val_acc: float, tag: str = 'best'):
        path = self.checkpoint_dir / f"{tag}_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_acc,
        }, path)
        return path

    def train(self):
        print(f"\n{'='*60}")
        print(f"Training on {self.device} | {self.epochs} epochs")
        print(f"Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_m = _run_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, self.scaler, is_train=True
            )
            val_loss, val_m = _run_epoch(
                self.model, self.val_loader, self.criterion,
                self.optimizer, self.device, self.scaler, is_train=False
            )

            self.scheduler.step()
            elapsed = time.time() - t0

            self._log(epoch, train_loss, train_m, val_loss, val_m, elapsed)

            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss, 'train_acc': train_m['accuracy'],
                'val_loss': val_loss,     'val_acc': val_m['accuracy'],
                **{f'val_{k}': v for k, v in val_m.items() if k != 'accuracy'},
            })

            val_acc = val_m['accuracy']
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                path = self._save_checkpoint(epoch, val_acc, tag='best')
                print(f"  ✓ New best — saved to {path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                    break

        # Save final checkpoint regardless
        self._save_checkpoint(self.epochs, self.best_val_acc, tag='last')
        print(f"\nTraining complete. Best val accuracy: {self.best_val_acc:.1f}%")
        return self.history
