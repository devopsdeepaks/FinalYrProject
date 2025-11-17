# Deepfake Detection System - Backend

## Project Overview

This is the backend implementation of a hybrid deepfake detection framework based on our research paper. The system integrates **EfficientNetB4**, **Attention Mechanisms**, and **Vision Transformers (ViTs)** to achieve robust deepfake detection with 92.4% accuracy on FaceForensics++.

### Key Features

- **Dual-Domain Preprocessing**: Fast Fourier Transform (FFT) + Local Binary Patterns (LBP)
- **Efficient Feature Extraction**: EfficientNetB4 with 38% parameter reduction through binarization
- **Attention-Based Localization**: 8-head spatial-channel attention for artifact detection
- **Global Context Modeling**: Vision Transformer with 16×16 patch embeddings
- **Real-time Inference**: 130ms latency on NVIDIA GPUs
- **Multi-Dataset Evaluation**: FaceForensics++, DFDC, COCOFake

## Project Structure

```
backend/
├── config.py                      # Configuration and hyperparameters
├── requirements.txt               # Python dependencies
│
├── preprocessing/                 # Preprocessing modules
│   ├── __init__.py
│   ├── fft_transform.py          # Fast Fourier Transform implementation
│   ├── lbp_features.py           # Local Binary Pattern features
│   ├── face_detector.py          # Face detection (MTCNN, Dlib, OpenCV)
│   └── preprocessor.py           # Main preprocessing pipeline
│
├── models/                        # Neural network models
│   ├── __init__.py
│   ├── efficientnet_model.py     # EfficientNetB4 backbone (✅ Completed)
│   ├── attention.py              # Attention mechanisms (🚧 In Progress)
│   ├── vision_transformer.py     # Vision Transformer (⏳ Pending)
│   └── ensemble_model.py         # Ensemble integration (⏳ Pending)
│
├── training/                      # Training pipeline
│   ├── trainer.py                # Training loop
│   ├── dataset.py                # Dataset loaders
│   └── augmentation.py           # Data augmentation
│
├── evaluation/                    # Evaluation metrics
│   ├── metrics.py                # Accuracy, F1, AUC-ROC
│   └── visualize.py              # Attention heatmaps, ROC curves
│
├── api/                          # REST API
│   ├── main.py                   # FastAPI application
│   └── routes.py                 # API endpoints
│
└── utils/                        # Utility functions
    ├── helpers.py
    └── logger.py
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM
- NVIDIA GPU (recommended for training)

### Setup Steps

1. **Clone the repository** (if not already done)

   ```bash
   cd /Users/krishkumar/Desktop/projects/finalYrProject
   ```

2. **Create virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Download pretrained models** (optional)
   ```bash
   # EfficientNetB4 will be automatically downloaded via timm
   # For face detection models:
   python -c "import dlib; dlib.get_frontal_face_detector()"
   ```

## Current Progress

### ✅ Completed Components

1. **Project Structure**: Complete folder organization
2. **Configuration**: Centralized config with all hyperparameters
3. **Preprocessing Modules**:
   - ✅ FFT Transform with frequency analysis
   - ✅ LBP Features with multi-scale support
   - ✅ Face Detection (MTCNN, Dlib, OpenCV, MediaPipe)
   - ✅ Comprehensive preprocessor pipeline
4. **EfficientNetB4 Model**:
   - ✅ Backbone architecture
   - ✅ Weight binarization (38% reduction)
   - ✅ MBConv blocks
   - ✅ Compound scaling
5. **Attention Mechanisms**:
   - ✅ Spatial attention
   - ✅ Channel attention
   - ✅ Combined spatial-channel attention
   - ✅ Multi-head self-attention (8 heads)
   - ✅ Attention heatmap generation

### 🚧 In Progress

- Vision Transformer implementation
- Ensemble model integration

### ⏳ Pending

- Training pipeline
- Evaluation metrics
- FastAPI REST API
- Model checkpoints and weights

## Research Paper Specifications

Our implementation follows these specifications from the research paper:

| Component               | Specification          |
| ----------------------- | ---------------------- |
| **Input Size**          | 224×224 RGB images     |
| **Patch Size**          | 16×16 (for ViT)        |
| **ViT Hidden Dim**      | 768                    |
| **ViT Layers**          | 12 transformer blocks  |
| **Attention Heads**     | 8 heads                |
| **Batch Size**          | 32                     |
| **Learning Rate**       | 1e-4                   |
| **Epochs**              | 20                     |
| **Parameter Reduction** | 38% (via binarization) |

### Target Performance

| Dataset             | Accuracy | F1-Score | AUC-ROC |
| ------------------- | -------- | -------- | ------- |
| **FaceForensics++** | 92.4%    | 0.912    | 0.751   |
| **DFDC**            | 89.7%    | 0.876    | 0.739   |
| **COCOFake**        | 87.4%    | 0.834    | 0.731   |

## Mathematical Foundations

### Fast Fourier Transform (FFT)

```
F_u(u,v) = Σ Σ f(a,b) * e^(-j2π(ua/Q + vb/R))
|F_u(u,v)| = sqrt(Re(F_u)^2 + Im(F_u)^2)
F_u_norm = (|F_u| - min|F_u|) / (max|F_u| - min|F_u|)
```

### Local Binary Pattern (LBP)

```
LBP = Σ(q=0 to q-1) s(i_q - i_c) * 2^q
where s(l) = 1 if l ≥ 0, else 0
LBP_norm = (LBP - min(LBP)) / (max(LBP) - min(LBP))
```

### Attention Mechanism

```
A = σ(f(W))  # Spatial attention
Y = W ⊙ A    # Element-wise multiplication
```

## Next Steps

To continue development, we'll implement:

1. **Vision Transformer** (next priority)

   - Patch embedding (16×16)
   - Positional encoding
   - 12 transformer blocks
   - Multi-head attention

2. **Ensemble Model**

   - Integrate EfficientNetB4 + Attention + ViT
   - Feature fusion strategies
   - Classification head

3. **Training Pipeline**

   - Dataset loaders for FaceForensics++, DFDC, COCOFake
   - Training loop with mixed precision
   - Learning rate scheduling
   - Model checkpointing

4. **API Development**
   - Image/video upload endpoints
   - Real-time inference
   - Attention heatmap visualization

## Usage Examples

### Preprocessing

```python
from preprocessing import DeepfakePreprocessor
import cv2

# Initialize preprocessor
preprocessor = DeepfakePreprocessor(
    image_size=224,
    use_fft=True,
    use_lbp=True,
    face_detection_method='mtcnn'
)

# Load and preprocess image
image = cv2.imread('test_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = preprocessor.preprocess(image_rgb)
# Returns: {'rgb': normalized_image, 'fft': fft_features, 'lbp': lbp_features}
```

### Model Usage (when complete)

```python
from models import DeepfakeDetectionModel
import torch

# Load model
model = DeepfakeDetectionModel(num_classes=2)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.softmax(output, dim=1)
    confidence = prediction[:, 1].item()  # Fake confidence
```

## Contributing

This is an academic research project. For questions or collaboration:

- Email: [your email]
- Research Paper: [link to paper]

## License

[Specify your license]

## Citations

If you use this code in your research, please cite our paper:

```bibtex
@article{yourpaper2025,
  title={Hybrid Deepfake Detection Framework Integrating EfficientNetB4, Attention Mechanisms, and Vision Transformers},
  author={Your Name et al.},
  journal={Conference/Journal Name},
  year={2025}
}
```

## Acknowledgments

- EfficientNet implementation from [timm](https://github.com/rwightman/pytorch-image-models)
- Face detection using [MTCNN](https://github.com/ipazc/mtcnn)
- Datasets: FaceForensics++, DFDC, COCOFake

---

**Status**: 🟢 Active Development | **Last Updated**: November 17, 2025
