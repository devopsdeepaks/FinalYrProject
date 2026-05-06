# DeepScan — Deepfake Detection System

A full-stack deepfake detection application powered by a hybrid **EfficientNetB4 + Spatial-Channel Attention** architecture, trained on the FaceForensics++ dataset.

## Live Demo

| Service | URL |
|---------|-----|
| Frontend | https://final-yr-project-brown.vercel.app |
| Backend API | https://finalyrproject-geoq.onrender.com |

---

## Architecture

### Model
- **Backbone** — EfficientNetB4 pretrained on ImageNet
- **Attention** — Spatial-channel feature attention module
- **Fusion** — Concatenation-based MLP classifier
- **Input** — 224×224 RGB, ImageNet normalisation
- **Output** — Binary (Real / Fake) with softmax confidence

### Stack
- **Backend** — FastAPI + PyTorch (CPU), deployed on Render
- **Frontend** — Next.js 16 + Tailwind CSS + Framer Motion, deployed on Vercel

---

## Repository Structure

```
├── backend/
│   ├── api/              # FastAPI app (main.py, routes.py)
│   ├── models/           # EfficientNetB4 backbone, attention, ensemble
│   ├── preprocessing/    # Face detection, FFT, LBP pipeline
│   ├── training/         # Dataset loader, transforms, trainer
│   ├── checkpoints/      # Saved model weights
│   ├── data/frames/      # Training data (real/ and fake/ frames)
│   ├── config.py         # All hyperparameters and paths
│   ├── quick_train_demo.py
│   └── requirements-minimal.txt
└── frontend/
    ├── app/
    │   ├── components/   # DeepfakeDetector, ResultCard, ConfidenceGauge, etc.
    │   ├── layout.tsx
    │   └── page.tsx
    └── package.json
```

---

## Running Locally

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements-minimal.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
# create .env.local with:
# NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:3000`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service status and model info |
| POST | `/predict` | Upload image, returns prediction |

### Example

```bash
curl -X POST https://finalyrproject-geoq.onrender.com/predict \
  -F "file=@photo.jpg"
```

```json
{
  "is_fake": true,
  "confidence": 0.934,
  "fake_probability": 0.934,
  "real_probability": 0.066,
  "face_detected": true
}
```

---

## Training

Training data is sourced from **FaceForensics++** (DeepFakes manipulation type). Frames are extracted from videos and stored under `backend/data/frames/real/` and `backend/data/frames/fake/`.

To retrain:

```bash
cd backend
source venv/bin/activate
python quick_train_demo.py
```

---

## Dataset Benchmarks

| Dataset | Accuracy |
|---------|----------|
| FaceForensics++ | 92.4% |
| DFDC | 89.7% |
| COCOFake | 87.4% |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![Vercel](https://img.shields.io/badge/Deployed-Vercel-black)
![Render](https://img.shields.io/badge/Backend-Render-purple)
