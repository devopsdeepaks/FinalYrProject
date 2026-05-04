# Complete Deployment Guide: Free Deepfake Detection Website

## Architecture Overview
```
Frontend (Next.js on Vercel) → API Call → Backend (FastAPI on Render/Railway)
                                               ↓
                                          ML Model
                                          (Prediction)
                                               ↓
                                         Send Result Back
```

---

## STEP 1: Train Your Model Locally

### 1.1 Train with 100 images
```bash
cd backend
python quick_train_demo.py
```

This will:
- Use your 50 fake + 50 real images
- Train for 10 epochs
- Save checkpoint to `backend/checkpoints/demo_model.pth`
- Takes ~15-30 minutes on CPU

### 1.2 Verify training completed
```bash
ls -lh backend/checkpoints/demo_model.pth
# Should see the .pth file with size ~100-200MB
```

---

## STEP 2: Deploy Backend (FREE - Using Render.com or Railway.app)

### Option A: Using Render.com (Recommended - Easiest)

#### 2A.1 Prepare backend for deployment

Create `backend/runtime.txt`:
```
python-3.10
```

#### 2A.2 Modify `backend/api/main.py` to accept CORS from frontend
Replace the CORS middleware section:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict later)
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 2A.3 Sign up & Deploy on Render.com
1. Go to [render.com](https://render.com)
2. Sign up (free)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository (or upload code)
5. Configure:
   - **Name**: `deepfake-detector-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements-minimal.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
6. Choose **Free Plan**
7. Deploy!

**Your backend URL will be**: `https://deepfake-detector-api.onrender.com`

#### 2A.4 Keep backend alive (optional for free tier)
Render spins down free services. Add to `backend/api/main.py`:
```python
import threading
import time

def keep_alive():
    """Keep the service alive by pinging itself periodically"""
    while True:
        try:
            time.sleep(600)  # Every 10 minutes
            # Ping the health endpoint
        except:
            pass

threading.Thread(target=keep_alive, daemon=True).start()
```

---

### Option B: Using Railway.app

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repo
5. Railway auto-detects Python and deploys
6. In Settings, set:
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
7. Get your deployed URL from Deployments

---

## STEP 3: Deploy Frontend (FREE - Using Vercel)

### 3.1 Push code to GitHub
```bash
cd /Users/deepak/Documents/explore/finalYrProject
git init
git add .
git commit -m "Initial commit: Deepfake detector with 100 images"
git remote add origin https://github.com/YOUR_USERNAME/finalYrProject.git
git branch -M main
git push -u origin main
```

### 3.2 Deploy on Vercel (Free)
1. Go to [vercel.com](https://vercel.com)
2. Sign in with GitHub
3. Click "Add New" → "Project"
4. Select your GitHub repo
5. Configure:
   - **Root Directory**: `frontend`
   - **Framework Preset**: `Next.js`
6. Add Environment Variable:
   - **Name**: `NEXT_PUBLIC_API_URL`
   - **Value**: `https://deepfake-detector-api.onrender.com` (your backend URL)
7. Click "Deploy"

**Your frontend URL will be**: `https://finalYrProject.vercel.app` (or custom domain)

---

## STEP 4: Connect Frontend to Backend

### 4.1 Update frontend API calls in `frontend/app/components/DeepfakeDetector.tsx`

```typescript
'use client';

import { useState, useRef } from 'react';
import Image from 'next/image';
import DropZone from './DropZone';
import ResultCard from './ResultCard';
import ConfidenceGauge from './ConfidenceGauge';
import ScanOverlay from './ScanOverlay';

export default function DeepfakeDetector() {
  const [image, setImage] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  const handleImageUpload = async (file: File) => {
    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
      setImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);

    // Send to backend
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to process image');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-black p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">DeepScan</h1>
          <p className="text-gray-300 text-lg">
            Detect deepfakes with AI-powered analysis
          </p>
        </div>

        {/* Main Content */}
        {!image ? (
          <DropZone onImageSelected={handleImageUpload} />
        ) : (
          <div className="space-y-6">
            {/* Image Preview */}
            <div className="bg-gray-800 rounded-lg p-6 relative">
              <div className="relative w-full h-96 bg-black rounded overflow-hidden">
                <Image
                  src={image}
                  alt="Uploaded"
                  fill
                  className="object-contain"
                />
                {loading && <ScanOverlay />}
              </div>
            </div>

            {/* Results */}
            {error && (
              <div className="bg-red-900 border border-red-600 text-white p-4 rounded-lg">
                Error: {error}
              </div>
            )}

            {result && !loading && (
              <div className="space-y-4">
                <ResultCard
                  label={result.predicted_label}
                  confidence={result.confidence}
                />
                <ConfidenceGauge confidence={result.confidence} />
              </div>
            )}

            {loading && (
              <div className="text-center text-white">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
                <p className="mt-4">Analyzing image...</p>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-4 justify-center">
              <button
                onClick={handleReset}
                className="px-8 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-semibold"
              >
                Clear
              </button>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold"
              >
                Upload Another
              </button>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleImageUpload(file);
              }}
              hidden
            />
          </div>
        )}
      </div>
    </div>
  );
}
```

### 4.2 Update `frontend/app/components/DropZone.tsx`

```typescript
'use client';

import { useState } from 'react';

interface DropZoneProps {
  onImageSelected: (file: File) => void;
}

export default function DropZone({ onImageSelected }: DropZoneProps) {
  const [isDragActive, setIsDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true);
    } else if (e.type === 'dragleave') {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      onImageSelected(files[0]);
    }
  };

  return (
    <div
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      className={`border-3 border-dashed rounded-2xl p-12 text-center transition-colors ${
        isDragActive
          ? 'border-blue-400 bg-blue-900/20'
          : 'border-gray-600 hover:border-gray-400 bg-gray-800/50'
      }`}
    >
      <div className="space-y-4">
        <svg
          className="mx-auto h-16 w-16 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 4v16m8-8H4"
          />
        </svg>
        <div>
          <p className="text-2xl font-bold text-white mb-2">
            Drag & Drop Image Here
          </p>
          <p className="text-gray-400 mb-4">or click to browse</p>
          <button
            onClick={() => document.querySelector('input[type="file"]')?.click()}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold"
          >
            Choose Image
          </button>
        </div>
        <p className="text-sm text-gray-500">
          JPG, PNG, WebP up to 10MB
        </p>
      </div>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onImageSelected(file);
        }}
        hidden
      />
    </div>
  );
}
```

---

## STEP 5: Test Everything

### 5.1 Test Backend Locally First
```bash
cd backend
python quick_train_demo.py  # Train model
python -m uvicorn api.main:app --reload
```

Then in another terminal:
```bash
curl -X POST -F "file=@sample_image.jpg" http://localhost:8000/predict
```

### 5.2 Test Frontend Locally
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` and upload an image.

---

## STEP 6: Deploy Complete Pipeline

### 6.1 Prepare `.env.local` for production (frontend)
```env
NEXT_PUBLIC_API_URL=https://deepfake-detector-api.onrender.com
```

### 6.2 Final Deployment
```bash
# Push to GitHub
git add .
git commit -m "Add live image detection API integration"
git push

# Vercel auto-deploys on push
# Render auto-deploys on push
```

---

## STEP 7: Monitor & Troubleshoot

### Check Backend Logs
- **Render**: Dashboard → "Logs" tab
- **Railway**: Dashboard → "Logs" tab

### Check Frontend Logs
- **Vercel**: Dashboard → "Deployments" → "Logs"
- Browser Console (F12)

### Common Issues

**Issue**: CORS errors
```
Access to XMLHttpRequest has been blocked by CORS policy
```
**Fix**: Ensure backend has CORS middleware configured

**Issue**: 404 on /predict
```
POST /predict returned 404
```
**Fix**: Verify FastAPI route is correctly defined in `api/routes.py`

**Issue**: Model not found
```
FileNotFoundError: demo_model.pth
```
**Fix**: Ensure model is trained and `.gitkeep` files exist in dirs

---

## STEP 8: Optimize for Free Tier

### Backend Optimization
```python
# In backend/config.py
GPU = False  # Keep CPU only for free tier
MAX_WORKERS = 2  # Limit concurrent requests
TIMEOUT = 30  # Auto-timeout after 30s
```

### Frontend Optimization
```javascript
// frontend/next.config.ts
export default {
  images: {
    unoptimized: true,
  },
  swcMinify: true,
}
```

---

## STEP 9: Custom Domain (Optional Free)

### Add custom domain to Vercel
1. Vercel Dashboard → Settings → Domains
2. Use free `.vercel.app` or connect custom domain via DNS

---

## Final Checklist

- [ ] Model trained with 100 images
- [ ] Backend deployed on Render/Railway
- [ ] Frontend deployed on Vercel
- [ ] Environment variable `NEXT_PUBLIC_API_URL` set
- [ ] CORS enabled on backend
- [ ] Image upload works end-to-end
- [ ] Results display correctly
- [ ] Repo on GitHub

---

## Costs: **$0** 🎉

| Service | Plan | Cost |
|---------|------|------|
| Vercel | Free | $0 |
| Render | Free | $0 |
| Railway | Free Credits | $0 |
| GitHub | Free | $0 |
| **Total** | | **$0** |

---

## Quick Start Commands

```bash
# 1. Train model
cd backend && python quick_train_demo.py

# 2. Test locally
python -m uvicorn api.main:app --reload  # Terminal 1
cd frontend && npm run dev               # Terminal 2

# 3. Push to GitHub
git add . && git commit -m "Deploy" && git push

# 4. Both services auto-deploy from GitHub!
```

That's it! Your live deepfake detector is live! 🚀
