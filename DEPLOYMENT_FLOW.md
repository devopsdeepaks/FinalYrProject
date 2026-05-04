# 📊 Deployment Flow Diagram

## Step-by-Step Visual Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ YOUR LOCAL COMPUTER (Now ✓ Complete)                             │
│                                                                   │
│  ✅ Model Trained: demo_model.pth (86.7% accuracy)              │
│  ✅ Backend Code: FastAPI with /predict endpoint                │
│  ✅ Frontend Code: Next.js React app                            │
│  ✅ Git Initialized: .gitignore, .gitattributes ready           │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ git push
             ↓
┌─────────────────────────────────────────────────────────────────┐
│ GITHUB (Free Repository)                                         │
│                                                                   │
│  - Stores your code                                              │
│  - Triggers automatic deployments                                │
└────┬────────────────────────────────────────────────────────────┘
     │
     ├─────────────────────┬─────────────────────┐
     │                     │                     │
     ↓                     ↓                     ↓
┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐
│ RENDER.COM   │  │ VERCEL.COM   │  │ GITHUB WORKFLOWS    │
│              │  │              │  │ (Optional Auto)     │
│ Backend:     │  │ Frontend:    │  │                     │
│ FastAPI App  │  │ Next.js App  │  │ Auto-deploy on push │
│              │  │              │  │                     │
│ /health      │  │ Upload Page  │  └─────────────────────┘
│ /predict ←──────────→ Calls API  │
│              │  │              │
│ Port 8000    │  │ Port 3000    │
└──────┬───────┘  └──────┬───────┘
       │                 │
       │                 │
       ↓                 ↓
   [ML Model]      [User Interface]
      (loads)      (displays results)
       │                 │
       └────────┬────────┘
                ↓
        🌐 LIVE ON INTERNET 🌐
        
https://deepfake-detector-api.onrender.com (Backend)
https://deepfake-detector.vercel.app (Frontend)
```

---

## Detailed Flow: What Happens When User Uploads Image

```
┌─────────────────────────────────────────────────────────────────┐
│ USER VISITS: https://deepfake-detector.vercel.app               │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │ 1. PAGE LOADS           │
        │    (Vercel Frontend)    │
        │ - React components      │
        │ - Tailwind CSS loaded   │
        │ - Ready for upload      │
        └────────────┬────────────┘
                     │
        ┌────────────┴────────────────────┐
        │ 2. USER UPLOADS IMAGE           │
        │    (Drag & Drop or Browse)      │
        │ - Image selected                │
        │ - Preview shown                 │
        │ - "Analyze" button active       │
        └────────────┬────────────────────┘
                     │
        ┌────────────┴────────────────────────┐
        │ 3. FRONTEND SENDS TO BACKEND        │
        │    POST /predict                    │
        │    Body: multipart/form-data        │
        │    Header: Content-Type: image/*    │
        │ URL: https://...onrender.com        │
        └────────────┬────────────────────────┘
                     │
        ┌────────────┴─────────────────────────┐
        │ 4. BACKEND PROCESSES (Render)        │
        │    - Loads image                     │
        │    - Face detection                 │
        │    - ML model inference             │
        │    - Calculate confidence           │
        │    - Return JSON response           │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────┴──────────────────┐
        │ 5. JSON RESPONSE SENT          │
        │ {                              │
        │   is_fake: true/false,         │
        │   confidence: 0.85,            │
        │   fake_probability: 0.85,      │
        │   real_probability: 0.15,      │
        │   face_detected: true          │
        │ }                              │
        └────────────┬──────────────────┘
                     │
        ┌────────────┴──────────────────┐
        │ 6. FRONTEND DISPLAYS RESULT    │
        │    - Large "FAKE" or "REAL"   │
        │    - Confidence gauge         │
        │    - Probability percentage   │
        │    - Option to upload another │
        └────────────┬──────────────────┘
                     │
                  ✅ DONE!
```

---

## File Locations (What Gets Deployed)

```
GitHub Repository: finalYrProject

├── backend/
│   ├── api/
│   │   ├── main.py          → Deployed to Render ✈️
│   │   └── routes.py        → Deployed to Render ✈️
│   ├── checkpoints/
│   │   └── demo_model.pth   → Deployed to Render ✈️ (Your trained model!)
│   ├── models/              → Deployed to Render ✈️
│   ├── preprocessing/       → Deployed to Render ✈️
│   ├── training/            → NOT deployed (only needed for training)
│   ├── requirements-minimal.txt → Deployed to Render ✈️
│   └── Dockerfile           → Used by Render ✈️
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx         → Deployed to Vercel ✈️
│   │   └── components/      → Deployed to Vercel ✈️
│   ├── next.config.ts       → Deployed to Vercel ✈️
│   ├── package.json         → Deployed to Vercel ✈️
│   └── Dockerfile           → NOT needed by Vercel (they use npm)
│
└── .gitignore              → Not deployed, used locally
```

---

## Timeline: What Happens After Each Step

### Step 1: Push to GitHub
```
git add .
git commit -m "Initial: Model trained on 100 images"
git push origin main
        │
        ├─ ✓ Code arrives on GitHub
        ├─ ✓ Render sees the push
        └─ ✓ Vercel sees the push
```

### Step 2: Deploy Backend (Render)
```
Render sees new commit
        ↓
    Starts build
        ├─ Clone repository
        ├─ Install Python 3.10
        ├─ Run: pip install -r requirements-minimal.txt
        ├─ Start: uvicorn api.main:app --host 0.0.0.0 --port 8000
        ├─ Test: curl /health
        └─ If OK: Service is LIVE ✅
        
Duration: ~2-3 minutes
You can watch in: Render Dashboard → Logs
```

### Step 3: Deploy Frontend (Vercel)
```
Vercel sees new commit
        ↓
    Starts build
        ├─ Clone repository
        ├─ Install Node.js
        ├─ Run: npm install (in frontend/)
        ├─ Run: npm run build
        ├─ Set env: NEXT_PUBLIC_API_URL=https://...onrender.com
        ├─ Optimize and deploy
        └─ If OK: Site is LIVE ✅
        
Duration: ~1-2 minutes
You can watch in: Vercel Dashboard → Deployments
```

### Step 4: Both Live!
```
Now you have:
  Backend: https://deepfake-detector-api.onrender.com
  Frontend: https://deepfake-detector.vercel.app
  
Both connected and working! 🎉
```

---

## What Each Service Does

### RENDER (Backend Hosting)
- Runs Python/FastAPI server
- Loads your ML model into memory
- Processes image predictions
- Free tier: Sleeps after 15 min inactivity (wakes on request)
- CPU only (no GPU on free)
- ~30s first request, ~2-3s subsequent

### VERCEL (Frontend Hosting)
- Serves Next.js React app
- Always running (no sleep)
- No cold start delays
- CDN for fast delivery
- ~1-2s page load

### GITHUB (Code Repository)
- Stores your code safely
- Webhooks trigger Render/Vercel deploys
- Free unlimited storage
- Version history (rollback if needed)

---

## Update Cycle After Deployment

```
You make changes locally:
    ↓
    git add .
    git commit -m "Fix bug / Add feature"
    git push origin main
        ↓
        GitHub receives code
        ├─ Webhook → Render
        │   ├─ Rebuild backend
        │   └─ Restart services
        │   (Takes ~2 min)
        │
        └─ Webhook → Vercel
            ├─ Rebuild frontend
            └─ Deploy new version
            (Takes ~1 min)
        
        Within 3 minutes, both services are updated!
        No manual steps needed!
```

---

## Real Example

### When you upload an image at `https://deepfake-detector.vercel.app`:

```
┌─────────────────────────────────────────┐
│ Your Browser (Vercel Frontend)           │
│ shows UI, sends HTTP POST                │
│ to https://deepfake-detector-api...     │
└────────────┬────────────────────────────┘
             │
             │ Image binary data
             │ (JPEG file bytes)
             │
             ↓
┌─────────────────────────────────────────┐
│ Render Server (Backend)                  │
│ - Gets image bytes                       │
│ - Decodes JPEG                           │
│ - Runs face detection                    │
│ - Sends to ML model                      │
│ - Gets predictions                       │
│ - Calculates confidence                  │
│ - Builds JSON response                   │
└────────────┬────────────────────────────┘
             │
             │ JSON response
             │ {is_fake: true, ...}
             │
             ↓
┌─────────────────────────────────────────┐
│ Your Browser (Vercel Frontend)           │
│ - Receives JSON                          │
│ - Updates state                          │
│ - Shows "FAKE: 87% confidence"          │
│ - Color changes (red for fake)          │
│ - Gauge fills up                        │
└────────────┬────────────────────────────┘
             │
          ✅ USER SEES RESULT!
```

---

## Cost Breakdown with Visual

```
┌─────────────────────────────────────┐
│ MONTHLY COST (Completely Free!)      │
├─────────────────────────────────────┤
│ Render Backend    │ $0/month         │
│ Vercel Frontend   │ $0/month         │
│ GitHub Storage    │ $0/month         │
│ Domain (optional) │ $0-12/year       │
├─────────────────────────────────────┤
│ TOTAL             │ $0 🎉           │
└─────────────────────────────────────┘
```

---

## Common Questions Answered

**Q: Why does first image take longer?**
A: Render free tier sleeps after 15 min. First request wakes it up (~10-30s). 
   Subsequent requests are fast (~2-3s).

**Q: Can I use this with my own domain?**
A: Yes! Vercel lets you connect custom domain for free. 
   See RENDER_VERCEL_SETUP.md Part 9.

**Q: What if I need GPU?**
A: Upgrade Render to paid tier ($7+/month with GPU).
   For now, CPU is sufficient for demos.

**Q: Can I update the model?**
A: Yes! Retrain locally, push to GitHub, Render auto-deploys with new model!

**Q: How many users can use simultaneously?**
A: Free tier can handle 1-3 concurrent requests.
   Good for demos. Upgrade for production.

**Q: Can I see who's using my site?**
A: Check Vercel analytics dashboard!
   Shows page views, regions, etc.

---

**Next Step: Push to GitHub and Deploy!** 🚀
See RENDER_VERCEL_SETUP.md for exact instructions.
