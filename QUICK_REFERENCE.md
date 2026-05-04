# 🚀 QUICK REFERENCE: Deploy Deepfake Detector for Free

## One-Minute Summary

```
Train Model → Test Locally → Push to GitHub → Deploy on Cloud → Live Site
```

---

## Command Cheatsheet

### 1. Train Model (10-30 min)
```bash
cd backend
python quick_train_demo.py
```

### 2. Test Backend Locally
```bash
cd backend
python -m uvicorn api.main:app --reload
```

In another terminal:
```bash
curl -X GET http://localhost:8000/health
curl -X POST -F "file=@samples/real/001.jpg" http://localhost:8000/predict
```

### 3. Test Frontend Locally
```bash
cd frontend
npm install
npm run dev
```
Visit: `http://localhost:3000`

### 4. Push to GitHub
```bash
git add .
git commit -m "Train model with 100 images"
git push origin main
```

### 5. Deploy Backend (Choose One)

**Render.com (Recommended)**
- Go to render.com
- Click "New Web Service"
- Connect GitHub repo
- Build: `pip install -r requirements-minimal.txt`
- Start: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Choose Free Plan
- Deploy!
- Get URL: `https://deepfake-detector-api.onrender.com`

**Railway.app**
- Go to railway.app
- Click "New Project"
- Connect GitHub repo
- Auto-detects Python
- Set Start Command: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Deploy!

### 6. Deploy Frontend (Vercel - Free)
- Go to vercel.com
- Click "Add New Project"
- Select GitHub repo
- Root Directory: `frontend`
- Add Environment Variable:
  - Name: `NEXT_PUBLIC_API_URL`
  - Value: `https://deepfake-detector-api.onrender.com`
- Deploy!
- Get URL: `https://your-project.vercel.app`

---

## File Structure

```
finalYrProject/
├── backend/
│   ├── api/
│   │   ├── main.py       ← FastAPI app
│   │   └── routes.py     ← /predict endpoint
│   ├── checkpoints/
│   │   └── demo_model.pth ← Trained model (generated after training)
│   ├── data/frames/
│   │   ├── fake/         ← 50 fake images
│   │   └── real/         ← 50 real images
│   ├── Dockerfile        ← Build backend image
│   ├── requirements-minimal.txt
│   └── quick_train_demo.py ← Run this to train
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx      ← Home page
│   │   └── components/
│   │       └── DeepfakeDetector.tsx ← Upload & detect
│   ├── Dockerfile        ← Build frontend image
│   ├── package.json
│   └── .env.local.example
│
├── docker-compose.yml    ← Run both services together
├── .gitignore           ← What git ignores
├── .gitattributes       ← Line endings & binary files
│
├── DEPLOYMENT_GUIDE.md  ← Full deployment walkthrough
├── TESTING_GUIDE.md     ← How to test API
└── quickstart.sh        ← Automated setup script
```

---

## API Endpoints

### Health Check
```
GET /health
Response: {status, device, mode, checkpoint}
```

### Prediction
```
POST /predict
Body: multipart/form-data with "file" field
Response: {is_fake, confidence, fake_probability, real_probability, face_detected}
```

### API Docs (when running)
```
http://localhost:8000/docs    (Swagger UI)
http://localhost:8000/redoc   (ReDoc)
```

---

## Deployment URLs

After deployment, you'll have:

| Service | URL |
|---------|-----|
| Frontend (Vercel) | https://your-project.vercel.app |
| Backend (Render) | https://deepfake-detector-api.onrender.com |
| API Docs | https://deepfake-detector-api.onrender.com/docs |

---

## Free Tier Limits

| Service | Free Tier |
|---------|-----------|
| Vercel | Unlimited deployments, 100GB bandwidth/month |
| Render | Sleep after 15 min inactivity, auto-wake |
| Railway | $5 credit/month |

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Model not found | Run `python quick_train_demo.py` in backend |
| Connection refused | Start backend: `uvicorn api.main:app --reload` |
| CORS error | Check `allow_origins=["*"]` in api/main.py |
| 404 /predict | Verify route in api/routes.py |
| Slow response | First request loads model (~3-5 sec) |

---

## Success Checklist

- [ ] Model trained (demo_model.pth exists)
- [ ] Backend tests pass (health endpoint works)
- [ ] Frontend loads at localhost:3000
- [ ] Can upload image and see prediction
- [ ] Code pushed to GitHub
- [ ] Backend deployed on Render/Railway
- [ ] Frontend deployed on Vercel
- [ ] Environment variable set in Vercel
- [ ] Live site works end-to-end
- [ ] Can upload images on live site

---

## Common Questions

**Q: Can I run backend and frontend together locally?**
A: Yes, with Docker Compose:
```bash
docker-compose up --build
```
Then visit `http://localhost:3000`

---

**Q: How long does training take?**
A: 10-30 minutes on CPU (100 images, 10 epochs)

---

**Q: Can I use more images?**
A: Yes! Update `MAX_PER_CLASS` in quick_train_demo.py
```python
MAX_PER_CLASS = 100  # Use 100 images per class (200 total)
```

---

**Q: Does the free tier have GPU?**
A: No, but it works fine with CPU for inference

---

**Q: What image formats are supported?**
A: JPG, PNG, WebP (any format PIL can read)

---

## Next Steps

1. Run training: `python quick_train_demo.py`
2. Test locally with guide: See `TESTING_GUIDE.md`
3. Deploy: Follow `DEPLOYMENT_GUIDE.md`
4. Share: Send frontend URL to anyone!

---

## Resources

- 📖 Full Guide: `DEPLOYMENT_GUIDE.md`
- 🧪 Testing: `TESTING_GUIDE.md`
- 🚀 Auto Setup: `bash quickstart.sh`
- 📝 Model Code: `backend/models/efficientnet_model.py`

---

## Cost Summary

| Service | Cost |
|---------|------|
| Domain (optional) | $0-12/year |
| Vercel hosting | $0 (free tier) |
| Render hosting | $0 (free tier) |
| Railway hosting | $0 (free tier) |
| GitHub | $0 |
| **TOTAL** | **$0** 🎉 |

---

**You're all set! Deploy with confidence! 🚀**
