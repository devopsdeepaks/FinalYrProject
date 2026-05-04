"""
Main training script for DeepScan deepfake detection.

Usage:
  # Train on downloaded Kaggle dataset (auto-detects path)
  python -m training.train --data_root ~/.cache/kagglehub/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset/versions/1

  # Override any setting
  python -m training.train --data_root /path/to/data --epochs 30 --batch_size 16

  # Evaluate only (requires best_model.pth checkpoint)
  python -m training.train --data_root /path/to/data --eval_only
"""

import argparse
import sys
from pathlib import Path

import torch

# Ensure backend root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _find_kaggle_dataset():
    """Try to find the auto-downloaded Kaggle dataset in the HuggingFace/Kaggle cache."""
    candidates = [
        Path.home() / ".cache/kagglehub/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset",
        Path.home() / ".cache/kaggle/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset",
    ]
    for c in candidates:
        if c.exists():
            # Find the deepest version directory that has files
            versions = sorted(c.glob("versions/*"), reverse=True)
            if versions:
                return str(versions[0])
            return str(c)
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepScan deepfake detector")

    # Data
    p.add_argument('--data_root', type=str, default=None,
                   help='Path to dataset root. Auto-detected if omitted.')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--frames_per_video', type=int, default=5,
                   help='Frames to sample per video (for video datasets)')
    p.add_argument('--train_ratio', type=float, default=0.70)
    p.add_argument('--val_ratio',   type=float, default=0.15)

    # Model
    p.add_argument('--backbone',    type=str, default='convnext_large',
                   choices=['efficientnet_b4', 'convnext_large', 'eva02_large'])
    p.add_argument('--vit_variant', type=str, default='pretrained_large',
                   choices=['custom', 'pretrained_large'])
    p.add_argument('--use_dino',    action='store_true', default=True)
    p.add_argument('--no_dino',     dest='use_dino', action='store_false')
    p.add_argument('--use_vit',     action='store_true', default=True)
    p.add_argument('--no_vit',      dest='use_vit',  action='store_false')
    p.add_argument('--fusion',      type=str, default='cross_attention',
                   choices=['concat', 'cross_attention'])

    # Training
    p.add_argument('--epochs',      type=int,   default=20)
    p.add_argument('--batch_size',  type=int,   default=16)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--weight_decay',type=float, default=1e-5)
    p.add_argument('--warmup_epochs',type=int,  default=2)
    p.add_argument('--patience',    type=int,   default=5,
                   help='Early stopping patience (epochs without improvement)')
    p.add_argument('--backbone_lr_ratio', type=float, default=0.1,
                   help='Backbone LR = base_lr * this ratio')
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--num_workers', type=int,   default=0)

    # Paths
    p.add_argument('--checkpoint_dir', type=str, default=None,
                   help='Where to save checkpoints (default: backend/checkpoints)')
    p.add_argument('--results_dir',    type=str, default=None,
                   help='Where to save evaluation JSON (default: backend/results)')

    # Mode
    p.add_argument('--eval_only', action='store_true',
                   help='Skip training, just evaluate best_model.pth on test set')
    p.add_argument('--resume',    type=str, default=None,
                   help='Path to checkpoint to resume from')

    return p.parse_args()


def main():
    args = parse_args()

    import config
    from models import DeepfakeDetectionModel
    from training.dataset import build_dataloaders
    from training.trainer import Trainer
    from training.evaluate import evaluate

    # ── Resolve paths ──────────────────────────────────────────────────────────
    data_root = args.data_root or _find_kaggle_dataset()
    if not data_root:
        print("ERROR: Could not find dataset. Pass --data_root /path/to/dataset")
        sys.exit(1)
    print(f"[Train] Dataset root: {data_root}")

    checkpoint_dir = Path(args.checkpoint_dir or config.CHECKPOINT_DIR)
    results_dir    = Path(args.results_dir    or config.RESULTS_DIR)

    # ── Device ────────────────────────────────────────────────────────────────
    device = config.DEVICE
    print(f"[Train] Device: {device}")

    torch.manual_seed(args.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        frames_per_video=args.frames_per_video,
        seed=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    model = DeepfakeDetectionModel(
        num_classes=2,
        image_size=args.image_size,
        dropout=config.MODEL_CONFIG['dropout'],
        use_attention=True,
        use_vit=args.use_vit and has_gpu,
        backbone_name=args.backbone,
        vit_variant=args.vit_variant,
        freeze_vit=False,
        use_dino=args.use_dino,
        freeze_dino=True,   # always freeze DINOv2 — only train fusion head
        use_clip=False,
        fusion=args.fusion,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[Train] Resumed from {args.resume}")

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        best_ckpt = checkpoint_dir / 'best_model.pth'
        if not best_ckpt.exists():
            print(f"ERROR: No checkpoint at {best_ckpt}. Train first.")
            sys.exit(1)
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        print(f"[Eval] Loaded checkpoint (val_acc={ckpt.get('val_accuracy', '?'):.1f}%)")
        evaluate(model, test_loader, device, results_dir)
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        device=device,
        epochs=args.epochs,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.patience,
        backbone_lr_ratio=args.backbone_lr_ratio,
    )
    trainer.train()

    # ── Final test-set evaluation ─────────────────────────────────────────────
    print("\n[Eval] Running test-set evaluation on best checkpoint...")
    best_ckpt = checkpoint_dir / 'best_model.pth'
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    evaluate(model, test_loader, device, results_dir)


if __name__ == '__main__':
    main()
