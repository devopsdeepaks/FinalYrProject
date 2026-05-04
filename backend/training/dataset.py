"""
Dataset loader for deepfake detection training.

Handles multiple folder layouts that Kaggle deepfake datasets use:

Layout A (flat):          Layout B (nested):        Layout C (per-method):
  root/                     root/                     root/
    real/                     train/real/               Deepfakes/
    fake/                     train/fake/               Original/
                              val/real/
                              val/fake/

Also handles video datasets by extracting frames on the fly.
Auto-detects which layout is present.
"""

import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ImageNet normalisation (matches pretrained backbone expectations)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomGrayscale(p=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])


def get_val_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── File discovery ────────────────────────────────────────────────────────────

def _find_files(folder: Path, extensions: set) -> list[Path]:
    return [p for p in folder.rglob('*') if p.suffix.lower() in extensions]


def _extract_frame(video_path: Path, frame_idx: int = 0) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _sample_video_frames(video_path: Path, num_frames: int = 5) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total == 0:
        return []
    indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
    frames = []
    for idx in indices:
        frame = _extract_frame(video_path, int(idx))
        if frame is not None:
            frames.append(frame)
    return frames


def _resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


# ── Dataset auto-detection ────────────────────────────────────────────────────

# Label names that indicate fake/real content (case-insensitive substring match)
_FAKE_KEYWORDS = {'fake', 'deepfake', 'manipulated', 'altered', 'synthetic',
                  'deepfakes', 'face2face', 'faceswap', 'neuraltextures', 'dfdc',
                  'dfd_manipulated'}
_REAL_KEYWORDS = {'real', 'original', 'authentic', 'genuine', 'youtube', 'actors',
                  'dfd_original'}


def _label_from_path(path: Path) -> Optional[int]:
    """
    Return 1=fake, 0=real by inspecting the immediate parent folders (last 3 parts).
    Only checks dataset-specific subdirectories, NOT the full absolute path,
    to avoid false matches on path components like 'deep-fake-detection-...' in
    the Kaggle cache directory name.
    """
    # Check only the last 3 path parts (direct parent folders + filename)
    for part in list(path.parts)[-3:]:
        low = part.lower()
        if any(k in low for k in _FAKE_KEYWORDS):
            return 1
        if any(k in low for k in _REAL_KEYWORDS):
            return 0
    return None


def discover_samples(root: Path, frames_per_video: int = 5) -> list[tuple[Path, int]]:
    """
    Walk root and return (file_path, label) pairs.
    Handles images directly; extracts representative frames from videos.
    Skips files where label cannot be determined from directory name.
    """
    samples: list[tuple[Path, int]] = []

    all_images = _find_files(root, IMAGE_EXTENSIONS)
    for p in all_images:
        label = _label_from_path(p)
        if label is not None:
            samples.append((p, label))

    all_videos = _find_files(root, VIDEO_EXTENSIONS)
    for v in all_videos:
        label = _label_from_path(v)
        if label is None:
            continue
        # Add video path N times — one random frame is extracted per __getitem__ call
        for _ in range(frames_per_video):
            samples.append((v, label))

    print(f"[Dataset] Discovered {len(samples)} samples in {root}")
    real = sum(1 for _, l in samples if l == 0)
    fake = sum(1 for _, l in samples if l == 1)
    print(f"[Dataset]   Real: {real}  Fake: {fake}")
    return samples


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake detection.
    Supports both image files and video files (random frame sampling).
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform=None,
        image_size: int = 224,
        frames_per_video: int = 1,
    ):
        self.samples = samples
        self.transform = transform
        self.image_size = image_size
        self.frames_per_video = frames_per_video

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        if path.suffix.lower() in VIDEO_EXTENSIONS:
            # Sample one random frame
            frame = _extract_frame(path, random.randint(0, 300))
            img = frame if frame is not None else np.zeros(
                (self.image_size, self.image_size, 3), dtype=np.uint8
            )
        else:
            raw = cv2.imread(str(path))
            if raw is None:
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        img = _resize_rgb(img, self.image_size)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, torch.tensor(label, dtype=torch.long)


# ── Train / Val / Test split helper ──────────────────────────────────────────

def split_samples(
    samples: list[tuple[Path, int]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    t = int(n * train_ratio)
    v = int(n * (train_ratio + val_ratio))
    return shuffled[:t], shuffled[t:v], shuffled[v:]


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_weighted_sampler(samples: list[tuple[Path, int]]):
    """
    Build a WeightedRandomSampler so each class is seen equally per epoch.
    Fixes the 8:1 fake/real imbalance in the DFD dataset.
    """
    from torch.utils.data import WeightedRandomSampler
    labels = [l for _, l in samples]
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    frames_per_video: int = 5,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(data_root)
    samples = discover_samples(root, frames_per_video=frames_per_video)

    if not samples:
        raise ValueError(f"No labelled samples found in {root}. "
                         "Ensure folders contain 'real'/'fake'/'original'/'manipulated' in their names.")

    train_s, val_s, test_s = split_samples(samples, train_ratio, val_ratio, seed)
    print(f"[Dataset] Split — train: {len(train_s)}  val: {len(val_s)}  test: {len(test_s)}")

    train_ds = DeepfakeDataset(train_s, get_train_transforms(image_size), image_size)
    val_ds   = DeepfakeDataset(val_s,   get_val_transforms(image_size),   image_size)
    test_ds  = DeepfakeDataset(test_s,  get_val_transforms(image_size),   image_size)

    # Weighted sampler on train set to balance 8:1 fake/real imbalance
    sampler = make_weighted_sampler(train_s)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_ds, sampler=sampler,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
