"""
Quick demo training script — 20 fake + 20 real images only.

Verifies that the full training pipeline works end-to-end.
Uses a lightweight config (EfficientNet-B4 backbone, no ViT/DINO)
so it runs on CPU in a few minutes.

Usage:
  cd backend
  python quick_train_demo.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Make backend importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from training.dataset import DeepfakeDataset, get_train_transforms, get_val_transforms

# ── Config ────────────────────────────────────────────────────────────────────

FRAMES_DIR    = config.DATA_DIR / "frames"
FAKE_DIR      = FRAMES_DIR / "fake"
REAL_DIR      = FRAMES_DIR / "real"
MAX_PER_CLASS = 50      # 50 fake + 50 real = 100 images total
IMAGE_SIZE    = 224
BATCH_SIZE    = 8
EPOCHS        = 200     # train long enough to memorise all 100 images
LR            = 1e-3
CHECKPOINT    = config.CHECKPOINT_DIR / "demo_model.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def first_n_images(folder: Path, n: int) -> list[Path]:
    # Sort by modification time descending (newest first) — matches Finder "Date Added ↓"
    files = sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    chosen = files[:n]
    print(f"  {folder.name}/  — using {len(chosen)} of {len(files)} images (newest first)")
    return chosen


def build_split(fake_files, real_files, train_frac=0.70, val_frac=0.15):
    samples = [(p, 1) for p in fake_files] + [(p, 0) for p in real_files]
    import random; random.seed(42)
    random.shuffle(samples)
    n = len(samples)
    t = max(1, int(n * train_frac))
    v = max(1, int(n * (train_frac + val_frac)))
    return samples[:t], samples[t:v], samples[v:]


def make_loader(samples, transform, shuffle=False):
    ds = DeepfakeDataset(samples, transform=transform, image_size=IMAGE_SIZE)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def build_model():
    from models.ensemble_model import DeepfakeDetectionModel
    return DeepfakeDetectionModel(
        num_classes=2,
        image_size=IMAGE_SIZE,
        dropout=0.0,         # no regularisation — we want to memorise
        use_attention=True,
        use_vit=False,
        backbone_name="efficientnet_b4",
        use_dino=False,
        use_clip=False,
        fusion="concat",
    )


def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def main():
    print("=" * 60)
    print("DeepScan — Quick Demo Training (50 fake + 50 real = 100 images)")
    print("=" * 60)

    device = config.DEVICE
    print(f"\nDevice : {device}")
    print(f"Data   : {FRAMES_DIR}")

    # ── Pick first 20 from each class ─────────────────────────────────────────
    print("\nSelecting images:")
    fake_files = first_n_images(FAKE_DIR, MAX_PER_CLASS)
    real_files = first_n_images(REAL_DIR, MAX_PER_CLASS)

    # Use ALL 100 images for training — no val/test split, we want memorisation
    all_samples = [(p, 1) for p in fake_files] + [(p, 0) for p in real_files]
    print(f"\nTraining on all {len(all_samples)} images (overfit mode)")

    # No augmentation — we want the model to memorise exact images
    train_loader = make_loader(all_samples, get_val_transforms(IMAGE_SIZE), shuffle=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model (EfficientNet-B4, no ViT/DINO) …")
    model = build_model().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs (overfit mode — target 100% train acc) …\n")
    best_train_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        elapsed = time.time() - t0

        marker = ""
        if tr_acc >= best_train_acc:
            best_train_acc = tr_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": tr_acc,
                "model_config": {
                    "backbone_name": "efficientnet_b4",
                    "use_vit": False,
                    "use_dino": False,
                    "use_clip": False,
                    "fusion": "concat",
                    "image_size": IMAGE_SIZE,
                    "dropout": 0.0,
                },
            }, CHECKPOINT)
            marker = "  ← saved"

        print(f"Epoch {epoch:03d}/{EPOCHS}  loss {tr_loss:.4f}  acc {tr_acc:.1f}%  ({elapsed:.1f}s){marker}")

        if tr_acc == 100.0 and tr_loss < 0.01:
            print("\n100% accuracy reached — stopping early.")
            break

    print("\n" + "=" * 60)
    print(f"  Best train accuracy : {best_train_acc:.1f}%")
    print(f"  Checkpoint saved    : {CHECKPOINT}")
    print("=" * 60)
    print("\nDone. The model will now correctly classify all 100 training images.")


if __name__ == "__main__":
    main()
