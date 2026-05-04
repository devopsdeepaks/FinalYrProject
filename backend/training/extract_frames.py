"""
Fast frame extractor using multiprocessing.Pool (bypasses GIL).
Extracts 5 evenly-spaced frames per video using sequential reads (no random seeking).

Usage:
  python -u training/extract_frames.py

Output: backend/data/frames/real/ and backend/data/frames/fake/
~7-10 minutes for the full DFD dataset.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_ROOT      = Path.home() / ".cache/kagglehub/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset/versions/1"
OUT_ROOT       = Path(__file__).resolve().parent.parent / "data" / "frames"
FRAMES_PER_VID = 5
VIDEO_EXTS     = {'.mp4', '.avi', '.mov', '.mkv'}
FAKE_KW        = {'manipulated', 'dfd_manipulated'}
REAL_KW        = {'original', 'dfd_original'}


def label_from_path(path: Path):
    for part in list(path.parts)[-3:]:
        low = part.lower()
        if any(k in low for k in FAKE_KW):
            return 'fake'
        if any(k in low for k in REAL_KW):
            return 'real'
    return None


def process_video(args):
    """Worker function — runs in a subprocess, no GIL contention."""
    video_path_str, out_dir_str, n_frames = args
    video_path = Path(video_path_str)
    out_dir    = Path(out_dir_str)

    cap   = cv2.VideoCapture(video_path_str)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return 0

    target_indices = set(np.linspace(0, total - 1, min(n_frames, total), dtype=int).tolist())
    stem  = video_path.stem
    saved = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in target_indices:
            slot = sorted(target_indices).index(frame_idx)
            out_path = out_dir / f"{stem}_f{slot:02d}.jpg"
            if not out_path.exists():
                frame_resized = cv2.resize(frame, (224, 224))
                cv2.imwrite(str(out_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
            if saved >= n_frames:
                break
        frame_idx += 1

    cap.release()
    return saved


def main():
    videos = [p for p in DATA_ROOT.rglob('*') if p.suffix.lower() in VIDEO_EXTS]
    print(f"Found {len(videos)} videos", flush=True)

    (OUT_ROOT / 'real').mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / 'fake').mkdir(parents=True, exist_ok=True)

    tasks = []
    for v in videos:
        label = label_from_path(v)
        if label:
            tasks.append((str(v), str(OUT_ROOT / label), FRAMES_PER_VID))

    print(f"Extracting {FRAMES_PER_VID} frames from {len(tasks)} labelled videos using {mp.cpu_count()} cores...", flush=True)

    total_frames = 0
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i, n in enumerate(pool.imap_unordered(process_video, tasks, chunksize=8), 1):
            total_frames += n
            if i % 200 == 0 or i == len(tasks):
                real_count = len(list((OUT_ROOT / 'real').glob('*.jpg')))
                fake_count = len(list((OUT_ROOT / 'fake').glob('*.jpg')))
                print(f"  [{i}/{len(tasks)}] real={real_count}  fake={fake_count}", flush=True)

    real = len(list((OUT_ROOT / 'real').glob('*.jpg')))
    fake = len(list((OUT_ROOT / 'fake').glob('*.jpg')))
    print(f"\nDone. Real frames: {real}  Fake frames: {fake}")
    print(f"Output: {OUT_ROOT}")


if __name__ == '__main__':
    main()
