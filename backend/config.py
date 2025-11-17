"""
Configuration file for the Deepfake Detection System
Contains all hyperparameters and settings from the research paper
"""

import os
from pathlib import Path

# Base Configuration
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for dir_path in [CHECKPOINT_DIR, LOGS_DIR, DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    "image_size": 224,
    "patch_size": 16,  # For Vision Transformer (16x16 patches)
    "num_classes": 2,  # Real vs Fake
    "efficientnet_version": "efficientnet_b4",
    "vit_hidden_dim": 768,
    "vit_num_layers": 12,
    "vit_num_heads": 8,
    "attention_heads": 8,
    "dropout": 0.1,
    "use_binarization": True,  # 38% parameter reduction
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "use_fft": True,
    "use_lbp": True,
    "fft_normalize": True,
    "lbp_radius": 1,
    "lbp_points": 8,
    "face_detection_method": "mtcnn",  # or "dlib", "opencv"
    "face_margin": 0.2,
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "optimizer": "adam",
    "scheduler": "cosine",
    "warmup_epochs": 2,
    "gradient_clip": 1.0,
    "early_stopping_patience": 5,
    "save_best_only": True,
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    "horizontal_flip_prob": 0.5,
    "rotation_limit": 15,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "blur_limit": 3,
    "gaussian_noise": 0.01,
}

# Dataset Configuration
DATASET_CONFIG = {
    "faceforensics++": {
        "path": DATA_DIR / "FaceForensics++",
        "compression": "c23",  # c0, c23, c40
        "manipulation_types": ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
    },
    "dfdc": {
        "path": DATA_DIR / "DFDC",
    },
    "cocofake": {
        "path": DATA_DIR / "COCOFake",
    },
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
}

# Inference Configuration
INFERENCE_CONFIG = {
    "batch_size": 1,
    "confidence_threshold": 0.5,
    "use_ensemble": True,
    "return_attention_maps": True,
    "device": "cuda",  # or "cpu"
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_upload_size": 100 * 1024 * 1024,  # 100 MB
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"],
    "temp_dir": BASE_DIR / "temp",
}

# Evaluation Metrics
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc_roc",
    "confusion_matrix",
]

# Performance Targets (from paper)
PERFORMANCE_TARGETS = {
    "faceforensics++": {
        "accuracy": 92.4,
        "f1_score": 0.912,
        "auc": 0.751,
    },
    "dfdc": {
        "accuracy": 89.7,
        "f1_score": 0.876,
        "auc": 0.739,
    },
    "cocofake": {
        "accuracy": 87.4,
        "f1_score": 0.834,
        "auc": 0.731,
    },
}

# Device Configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

# Random Seed for Reproducibility
SEED = 42
