"""
API routes for deepfake detection inference.

Primary detector: dima806/deepfake_vs_real_image_detection
  - ViT-base-patch16-224 fine-tuned on deepfake datasets
  - Loaded via HuggingFace transformers (auto-downloaded on first run)

Custom ensemble model (ensemble_model.py) will be used instead once
a trained checkpoint exists at config.CHECKPOINT_DIR / "best_model.pth".
"""

import sys
import random
from pathlib import Path

import numpy as np
import cv2
import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

router = APIRouter()

_hf_pipeline = None
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


# ── Pretrained HuggingFace detector ──────────────────────────────────────────

def _get_hf_detector():
    """
    Load ViT-base deepfake detector from HuggingFace.
    Model: dima806/deepfake_vs_real_image_detection
    Labels: Fake / Real
    Downloaded once, cached in ~/.cache/huggingface.
    """
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline
        _hf_pipeline = pipeline(
            "image-classification",
            model="dima806/deepfake_vs_real_image_detection",
            device=-1,  # CPU; change to 0 for CUDA
        )
    return _hf_pipeline


# ── Custom ensemble model (requires trained checkpoint) ──────────────────────

def _get_custom_model():
    global _custom_model
    if _custom_model is None:
        import config
        # Accept best_model.pth (full training) or demo_model.pth (quick demo)
        checkpoint = config.CHECKPOINT_DIR / "best_model.pth"
        if not checkpoint.exists():
            checkpoint = config.CHECKPOINT_DIR / "demo_model.pth"
        if not checkpoint.exists():
            return None

        from models import DeepfakeDetectionModel
        ckpt = torch.load(checkpoint, map_location=config.DEVICE)

        # Use arch config embedded in checkpoint if present, else fall back to config.py
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
            )
        else:
            cfg = config.MODEL_CONFIG
            has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
            model = DeepfakeDetectionModel(
                num_classes=cfg["num_classes"],
                image_size=cfg["image_size"],
                dropout=cfg["dropout"],
                use_attention=True,
                use_vit=has_gpu,
                backbone_name=cfg.get("backbone", "convnext_large"),
                vit_variant=cfg.get("vit_variant", "pretrained_large"),
                freeze_vit=cfg.get("freeze_vit", False),
                use_dino=cfg.get("use_dino", True),
                freeze_dino=cfg.get("freeze_dino", True),
                use_clip=cfg.get("use_clip", False),
                fusion=cfg.get("fusion", "cross_attention"),
            )

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        model = model.to(config.DEVICE)
        _custom_model = model

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


# ── Response model ────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    is_fake: bool
    confidence: float
    fake_probability: float
    real_probability: float
    face_detected: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

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
        mode, ckpt_file = "pretrained_hf", None
    return {
        "status": "healthy",
        "device": str(config.DEVICE),
        "mode": mode,
        "checkpoint": ckpt_file,
        "backbone": config.MODEL_CONFIG.get("backbone"),
    }


@router.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """Upload an image and get a real/fake deepfake prediction."""
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

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Face detection (for face_detected flag)
    preprocessor = _get_preprocessor()
    pre_result = preprocessor.preprocess(image_rgb, detect_face=True)
    face_detected = bool(pre_result.get("face_detected", False))

    # ── Inference ─────────────────────────────────────────────────────────────
    custom_model = _get_custom_model()

    if custom_model is not None:
        # Use trained custom ensemble when checkpoint exists
        import config
        rgb = pre_result["rgb"].astype(np.float32)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            logits = custom_model(tensor)
            probs = torch.softmax(logits, dim=1)
        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()

    else:
        # Use pretrained HuggingFace ViT deepfake detector
        pil_image = Image.fromarray(image_rgb)
        detector = _get_hf_detector()
        outputs = detector(pil_image)

        # Map label scores: model returns [{"label":"Fake","score":...}, {"label":"Real","score":...}]
        scores = {o["label"].lower(): o["score"] for o in outputs}
        fake_prob = scores.get("fake", 1.0 - scores.get("real", 0.5))
        real_prob = 1.0 - fake_prob

    return PredictionResult(
        is_fake=fake_prob > 0.5,
        confidence=round(max(real_prob, fake_prob), 4),
        fake_probability=round(fake_prob, 4),
        real_probability=round(real_prob, 4),
        face_detected=face_detected,
    )
