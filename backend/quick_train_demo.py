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
EPOCHS        = 10
LR            = 3e-4
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
        dropout=0.1,
        use_attention=True,
        use_vit=False,       # too heavy for a 40-image CPU demo
        backbone_name="efficientnet_b4",
        use_dino=False,      # same — skip large pretrained download
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

    train_s, val_s, test_s = build_split(fake_files, real_files)
    print(f"\nSplit  : train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")

    train_loader = make_loader(train_s, get_train_transforms(IMAGE_SIZE), shuffle=True)
    val_loader   = make_loader(val_s,   get_val_transforms(IMAGE_SIZE))
    test_loader  = make_loader(test_s,  get_val_transforms(IMAGE_SIZE))

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model (EfficientNet-B4, no ViT/DINO) …")
    model = build_model().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs …\n")
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, False)
        elapsed = time.time() - t0

        marker = ""
        if vl_acc >= best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": vl_acc,
                # store arch so the API can load with the correct config
                "model_config": {
                    "backbone_name": "efficientnet_b4",
                    "use_vit": False,
                    "use_dino": False,
                    "use_clip": False,
                    "fusion": "concat",
                    "image_size": IMAGE_SIZE,
                    "dropout": 0.1,
                },
            }, CHECKPOINT)
            marker = "  ← saved"

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train loss {tr_loss:.4f} acc {tr_acc:.1f}%  |  "
              f"val loss {vl_loss:.4f} acc {vl_acc:.1f}%  "
              f"({elapsed:.1f}s){marker}")

    # ── Test evaluation ────────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint ({CHECKPOINT}) …")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, False)

    print("\n" + "=" * 60)
    print(f"  Best val accuracy : {best_val_acc:.1f}%")
    print(f"  Test accuracy     : {test_acc:.1f}%")
    print(f"  Checkpoint saved  : {CHECKPOINT}")
    print("=" * 60)
    print("\nDemo training complete. The pipeline is working correctly.")


if __name__ == "__main__":
    main()
