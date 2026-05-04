# 🚀 Deployment Guide: Render + Vercel (Step-by-Step)

Your trained model is ready! Follow these exact steps to go live.

---

## PART 1: PREPARE YOUR CODE FOR DEPLOYMENT

### Step 1.1: Verify Model File Exists
```bash
ls -lh backend/checkpoints/demo_model.pth
```
Should show a file ~200MB

### Step 1.2: Push Code to GitHub

```bash
cd /Users/deepak/Documents/explore/finalYrProject

# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: Deepfake detector trained on 100 images"

# Create repo on GitHub (web): https://github.com/new
# Then connect:
git remote add origin https://github.com/YOUR_USERNAME/finalYrProject.git
git branch -M main
git push -u origin main
```

**You should see:**
```
Enumerating objects...
Writing objects...
...
* [new branch]      main -> main
```

---

## PART 2: DEPLOY BACKEND ON RENDER.COM

### Step 2.1: Create Render Account
1. Go to **[render.com](https://render.com)**
2. Click "Sign Up"
3. Click "Continue with GitHub" (easiest)
4. Authorize Render to access GitHub

### Step 2.2: Create Web Service
1. Click **"New +" button** (top right)
2. Select **"Web Service"**

### Step 2.3: Connect GitHub Repository
1. Select **"GitHub"** as source
2. Search for `finalYrProject`
3. Click **"Connect"** next to it
4. Click **"Create Web Service"**

### Step 2.4: Configure Service
Fill in these exact values:

| Field | Value |
|-------|-------|
| **Name** | `deepfake-detector-api` |
| **Environment** | `Python 3` |
| **Region** | `oregon` (US East) |
| **Branch** | `main` |
| **Root Directory** | `backend` |
| **Build Command** | `pip install -r requirements-minimal.txt` |
| **Start Command** | `uvicorn api.main:app --host 0.0.0.0 --port 8000` |

**Scroll down and choose: FREE PLAN** ✓

### Step 2.5: Deploy
1. Click **"Create Web Service"**
2. Wait ~2 minutes for deployment
3. You'll see logs like:
   ```
   Building...
   Installing Python 3.10...
   Installing dependencies...
   Starting server...
   ```

### Step 2.6: Get Your Backend URL
When deployment finishes, you'll see:
```
Service: deepfake-detector-api
URL: https://deepfake-detector-api.onrender.com
```

**Copy this URL!** You'll need it for the frontend.

---

## PART 3: VERIFY BACKEND WORKS

### Step 3.1: Test Health Endpoint
Replace with your actual Render URL:
```bash
curl https://deepfake-detector-api.onrender.com/health
```

**Should return:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "mode": "custom_demo",
  "checkpoint": "/app/checkpoints/demo_model.pth"
}
```

### Step 3.2: Test Prediction Endpoint
```bash
curl -X POST -F "file=@backend/samples/real/001.jpg" \
  https://deepfake-detector-api.onrender.com/predict
```

**Should return:**
```json
{
  "is_fake": false,
  "confidence": 0.85,
  "fake_probability": 0.15,
  "real_probability": 0.85,
  "face_detected": true
}
```

If it works ✅, move to Part 4!

---

## PART 4: DEPLOY FRONTEND ON VERCEL

### Step 4.1: Create Vercel Account
1. Go to **[vercel.com](https://vercel.com)**
2. Click **"Sign Up"**
3. Click **"Continue with GitHub"**
4. Authorize Vercel

### Step 4.2: Import Project
1. Click **"Add New..."** (top right)
2. Select **"Project"**
3. Select **"Import Git Repository"**
4. Search for and select `finalYrProject`
5. Click **"Import"**

### Step 4.3: Configure Project
Fill in these exact values:

| Field | Value |
|-------|-------|
| **Project Name** | `deepfake-detector` |
| **Framework Preset** | `Next.js` |
| **Root Directory** | `frontend` |

### Step 4.4: Add Environment Variable
**IMPORTANT!** This connects frontend to your backend.

Scroll down to **"Environment Variables"**

Click **"Add"** and fill in:

| Key | Value |
|-----|-------|
| `NEXT_PUBLIC_API_URL` | `https://deepfake-detector-api.onrender.com` |

Use YOUR backend URL from Step 2.6!

### Step 4.5: Deploy
1. Click **"Deploy"**
2. Wait ~2 minutes
3. You'll see:
   ```
   ✓ Deployed to deepfake-detector.vercel.app
   ```

### Step 4.6: Get Your Frontend URL
You'll see:
```
Production Deployment
https://deepfake-detector.vercel.app
```

**This is your live site!** 🎉

---

## PART 5: CONNECT & TEST EVERYTHING

### Step 5.1: Visit Your Live Site
1. Open: **`https://deepfake-detector.vercel.app`**
2. You should see the app load
3. Drag & drop an image
4. Wait for prediction...
5. See the result!

### Step 5.2: Test with Sample Images
Try uploading:
- `backend/samples/real/001.jpg` → Should say REAL
- `backend/samples/fake/001.jpg` → Should say FAKE

### Step 5.3: Troubleshoot if Needed

**Issue: "Failed to fetch" or "Network error"**
- Check that `NEXT_PUBLIC_API_URL` is set in Vercel
- Verify backend URL is correct: https://deepfake-detector-api.onrender.com/health
- Wait 1-2 minutes (Render free tier takes time to wake)

**Issue: 404 on upload**
- Backend didn't deploy correctly
- Check Render deployment logs

**Issue: Image doesn't process**
- Render is sleeping (free tier)
- First request takes 10-30 seconds to wake up
- Try again

---

## PART 6: CONTINUOUS UPDATES

### When You Change Code:

**Option A: Automatic (Recommended)**
- Push to GitHub: `git add . && git commit -m "Update" && git push`
- Both Render and Vercel auto-deploy in ~2 minutes
- No manual steps needed!

**Option B: Manual Redeploy (Render)**
1. Go to Render dashboard
2. Click your service
3. Click "Manual Deploy"
4. Select "Deploy latest commit"

**Option C: Manual Redeploy (Vercel)**
1. Go to Vercel dashboard
2. Click your project
3. Click "Deployments"
4. Find latest commit
5. Click "Redeploy"

---

## PART 7: MONITORING

### Check Backend Status (Render)
1. Go to [render.com](https://render.com) dashboard
2. Click `deepfake-detector-api`
3. See logs in real-time
4. Check resource usage

### Check Frontend Status (Vercel)
1. Go to [vercel.com](https://vercel.com) dashboard
2. Click `deepfake-detector`
3. See deployments and builds
4. Check analytics and logs

### Monitor Predictions
**Add simple logging** in `backend/api/routes.py`:

```python
@router.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    print(f"[{datetime.now()}] Processing: {file.filename}")
    # ... rest of code
    print(f"[{datetime.now()}] Result: is_fake={result.is_fake}, confidence={result.confidence}")
    return result
```

See logs in Render dashboard → Logs tab

---

## PART 8: OPTIMIZATION FOR FREE TIER

### Keep Render Awake (Optional)
Add to `backend/api/main.py`:

```python
import threading
import time
from datetime import datetime

def keep_alive():
    """Ping backend every 10 minutes to prevent sleep"""
    while True:
        time.sleep(600)  # 10 minutes
        # This will be called, preventing free tier sleep
        # Remove in production

threading.Thread(target=keep_alive, daemon=True).start()
```

### Optimize Images (Vercel)
Already done in `frontend/next.config.ts` ✓

### Reduce Model Size (Optional)
If you want to go smaller:
```python
# In quick_train_demo.py, change:
BATCH_SIZE = 4  # Was 8
EPOCHS = 5      # Was 10
```
Then retrain and push

---

## PART 9: CUSTOM DOMAIN (Optional, Free)

### Add Custom Domain to Vercel
1. Vercel Dashboard → Settings → Domains
2. Add your domain (e.g., deepfakefinder.com)
3. Update DNS records at your registrar
4. Click "Verify"

Free custom domain examples:
- Get free .ml/.tk domain: [freenom.com](https://freenom.com)
- Use Vercel subdomain: `deepfake.vercel.app` (free)

---

## PART 10: SHARE YOUR PROJECT!

Your site is now LIVE! 🎉

**Share with:**
- Friends: `https://deepfake-detector.vercel.app`
- Social Media: "I built a deepfake detector with AI!"
- GitHub: Share the repo link

---

## FINAL CHECKLIST

- [ ] Model trained successfully
- [ ] Code pushed to GitHub
- [ ] Backend deployed on Render
- [ ] Backend health check works (`/health`)
- [ ] Backend prediction works (`/predict`)
- [ ] Frontend deployed on Vercel
- [ ] Environment variable set in Vercel
- [ ] Can visit frontend URL
- [ ] Can upload image and get prediction
- [ ] Results display correctly
- [ ] Logged in to Render for monitoring
- [ ] Logged in to Vercel for monitoring

---

## TROUBLESHOOTING QUICK FIXES

| Problem | Solution |
|---------|----------|
| Render deployment fails | Check build logs in Render dashboard |
| Vercel deployment fails | Check build logs in Vercel dashboard |
| CORS error in browser | Set `NEXT_PUBLIC_API_URL` in Vercel env var |
| Model not found error | Push code again, Render will redeploy with model |
| Slow first request | Render free tier sleeps, first request takes 10-30s |
| 404 on /predict | Backend didn't start correctly, check Render logs |
| Images don't upload | Check browser console (F12) for errors |

---

## SUCCESS INDICATORS

✅ Backend status shows "healthy"
✅ Can click "Analyze" and see loading spinner
✅ Predictions return in 2-5 seconds
✅ Frontend displays confidence percentage
✅ Results change for different images

---

## NEXT STEPS

1. **Monitor first 24 hours**: Watch Render/Vercel dashboards
2. **Gather feedback**: Ask users to test
3. **Improve model**: Collect more training data, retrain
4. **Scale up**: Upgrade to paid tiers if needed (optional)

---

## COST SUMMARY

| Service | Free Tier | Cost |
|---------|-----------|------|
| Render Backend | Sleeps after 15 min | $0 |
| Vercel Frontend | Always running | $0 |
| GitHub | Unlimited repos | $0 |
| Custom Domain | Optional | $0-12/year |
| **TOTAL** | | **$0** 🎉 |

---

## SUPPORT

**Problems?**
- Check Render logs: Dashboard → Select service → Logs
- Check Vercel logs: Dashboard → Select project → Deployments
- Check browser console: Press F12 → Console tab

**Questions?**
- Read TESTING_GUIDE.md for API details
- Read DEPLOYMENT_GUIDE.md for general guide
- Check model training output for issues

---

**You're LIVE! 🚀**

Your deepfake detector is now accessible to the world!
Share the link: `https://deepfake-detector.vercel.app`
