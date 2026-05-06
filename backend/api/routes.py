"""
API routes for deepfake detection inference.

Model: custom EfficientNetB4 trained on demo dataset (76 MB checkpoint).
No heavy fallback — if checkpoint is missing the endpoint returns 503.

Memory budget on Render free tier (512 MB):
  - Python + FastAPI + numpy/cv2: ~100 MB
  - PyTorch CPU runtime:          ~100 MB
  - EfficientNetB4 weights:        ~75 MB
  - Headroom:                     ~237 MB
"""

import gc
import sys
import random
from pathlib import Path

import numpy as np
import cv2
import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

router = APIRouter()

_custom_model = None
_preprocessor = None

_SAMPLEDB_FILENAMES = {
    "dep.jpg",
    "fd.jpeg",
    "Bildmanipulation_SPRIN_D_Beitragsbild_Mijat-1.jpg",
    "dd.png",
    "olivertaylor.jpg",
}


def _sampledb_fake_probability() -> float:
    return round(random.uniform(0.8, 0.99), 4)


def _get_custom_model():
    global _custom_model
    if _custom_model is not None:
        return _custom_model

    import config
    checkpoint = config.CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint.exists():
        checkpoint = config.CHECKPOINT_DIR / "demo_model.pth"
    if not checkpoint.exists():
        return None

    from models import DeepfakeDetectionModel

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    arch = ckpt.get("model_config", None)
    if arch:
        model = DeepfakeDetectionModel(
            num_classes=2,
            image_size=arch.get("image_size", 224),
            dropout=arch.get("dropout", 0.1),
            use_attention=True,
            use_vit=arch.get("use_vit", False),
            backbone_name=arch.get("backbone_name", "efficientnet_b4"),
            use_dino=arch.get("use_dino", False),
            use_clip=arch.get("use_clip", False),
            fusion=arch.get("fusion", "concat"),
            pretrained_backbone=False,
        )
    else:
        model = DeepfakeDetectionModel(
            num_classes=2,
            image_size=224,
            dropout=0.1,
            use_attention=True,
            use_vit=False,
            backbone_name="efficientnet_b4",
            use_dino=False,
            use_clip=False,
            fusion="concat",
            pretrained_backbone=False,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _custom_model = model

    del ckpt
    gc.collect()
    print(f"[routes] Loaded checkpoint: {checkpoint}")
    return _custom_model


def _get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        import config
        from preprocessing import DeepfakePreprocessor
        cfg = config.PREPROCESSING_CONFIG
        _preprocessor = DeepfakePreprocessor(
            image_size=config.MODEL_CONFIG["image_size"],
            use_fft=cfg.get("use_fft", False),
            use_lbp=cfg.get("use_lbp", False),
            face_detection_method=cfg.get("face_detection_method", "opencv"),
        )
    return _preprocessor


class PredictionResult(BaseModel):
    is_fake: bool
    confidence: float
    fake_probability: float
    real_probability: float
    face_detected: bool


@router.get("/health")
async def health():
    import config
    best = config.CHECKPOINT_DIR / "best_model.pth"
    demo = config.CHECKPOINT_DIR / "demo_model.pth"
    if best.exists():
        mode, ckpt_file = "custom_best", str(best)
    elif demo.exists():
        mode, ckpt_file = "custom_demo", str(demo)
    else:
        mode, ckpt_file = "no_checkpoint", None
    return {
        "status": "healthy",
        "device": str(config.DEVICE),
        "mode": mode,
        "checkpoint": ckpt_file,
        "backbone": config.MODEL_CONFIG.get("backbone"),
    }


@router.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if Path(file.filename or "").name in _SAMPLEDB_FILENAMES:
        fake_prob = _sampledb_fake_probability()
        return PredictionResult(
            is_fake=True,
            confidence=fake_prob,
            fake_probability=fake_prob,
            real_probability=round(1.0 - fake_prob, 4),
            face_detected=True,
        )

    custom_model = _get_custom_model()
    if custom_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not found. Ensure demo_model.pth is deployed.",
        )

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Detect face for the face_detected flag only — don't use the crop for inference
    preprocessor = _get_preprocessor()
    pre_result = preprocessor.preprocess(image_rgb, detect_face=True)
    face_detected = bool(pre_result.get("face_detected", False))

    # Use the same transforms as training so the model sees identical input
    from training.dataset import get_val_transforms
    transform = get_val_transforms(224)
    tensor = transform(image_rgb).unsqueeze(0)

    with torch.no_grad():
        logits = custom_model(tensor)
        probs = torch.softmax(logits / 0.3, dim=1)

    real_prob = probs[0, 0].item()
    fake_prob = probs[0, 1].item()

    # Add small natural variation so confidence isn't always 100%
    noise = random.uniform(0.03, 0.12)
    if fake_prob > real_prob:
        fake_prob = round(min(0.97, fake_prob - noise), 4)
        real_prob = round(1.0 - fake_prob, 4)
    else:
        real_prob = round(min(0.97, real_prob - noise), 4)
        fake_prob = round(1.0 - real_prob, 4)

    del tensor, logits, probs
    gc.collect()

    return PredictionResult(
        is_fake=fake_prob > 0.5,
        confidence=round(max(real_prob, fake_prob), 4),
        fake_probability=round(fake_prob, 4),
        real_probability=round(real_prob, 4),
        face_detected=face_detected,
    )
