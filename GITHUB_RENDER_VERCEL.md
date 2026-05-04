# 📤 Push to GitHub & Deploy (Final Steps)

## STEP 1: Create GitHub Repository

### 1.1 Create on GitHub Website
1. Go to **[github.com/new](https://github.com/new)**
2. Fill in:
   - **Repository name**: `finalYrProject` (or any name)
   - **Description**: "AI Deepfake Detector - Live on Render & Vercel"
   - **Public** (so Render/Vercel can access)
   - Do NOT initialize with README/license/gitignore (we have these)
3. Click **"Create Repository"**

You'll see instructions. **Copy your repo URL** (looks like):
```
https://github.com/YOUR_USERNAME/finalYrProject.git
```

---

## STEP 2: Push Your Code to GitHub

### 2.1 Open Terminal
```bash
cd /Users/deepak/Documents/explore/finalYrProject
```

### 2.2 Initialize Git (if not done)
```bash
git init
```

### 2.3 Add All Files
```bash
git add .
```

You should see output like:
```
adding backend/api/main.py
adding backend/checkpoints/demo_model.pth
adding frontend/app/page.tsx
...
```

### 2.4 Create Initial Commit
```bash
git commit -m "Initial commit: Deepfake detector trained on 100 images with 86.7% accuracy"
```

Output:
```
[master (root-commit) abc1234] Initial commit...
 150 files changed, 50+ insertions(+)
 create mode 100644 backend/...
 ...
```

### 2.5 Add Remote Repository
Replace `YOUR_USERNAME` with your actual GitHub username:
```bash
git remote add origin https://github.com/YOUR_USERNAME/finalYrProject.git
```

### 2.6 Rename Branch to Main
```bash
git branch -M main
```

### 2.7 Push to GitHub
```bash
git push -u origin main
```

**You'll be asked for credentials:**
- Option A: GitHub Personal Access Token (recommended)
  - Go to [github.com/settings/tokens](https://github.com/settings/tokens)
  - Click "Generate new token (classic)"
  - Select: repo, workflow
  - Copy token and paste when prompted
  
- Option B: GitHub Password (deprecated but may work)
  - Just paste your password when prompted

**Output when successful:**
```
Enumerating objects: 150, done.
Counting objects: 100% (150/150), done.
Delta compression using up to 8 threads
Compressing objects: 100% (120/120), done.
Writing objects: 100% (150/150), 150 MiB
...
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

✅ **Your code is now on GitHub!**

---

## STEP 3: Verify GitHub

### 3.1 Check Your Repo
1. Go to `https://github.com/YOUR_USERNAME/finalYrProject`
2. You should see:
   - All your files
   - `demo_model.pth` in `backend/checkpoints/`
   - `.gitignore` and `.gitattributes`
   - README files

### 3.2 Verify Model File is Included
- Click on `backend/checkpoints/`
- Should see `demo_model.pth` (~200MB)

---

## STEP 4: Deploy Backend on Render

### 4.1 Go to Render
1. **[render.com](https://render.com)**
2. Click **"Sign Up"** → **"Continue with GitHub"**
3. Authorize Render to access your GitHub account

### 4.2 Create Web Service
1. Click **"New +"** (top right)
2. Select **"Web Service"**
3. Choose **"GitHub"** as source
4. Search: `finalYrProject`
5. Click **"Connect"** next to it

### 4.3 Configure Service
**Copy these exact values:**

```
Name:                 deepfake-detector-api
Environment:          Python 3
Region:               Oregon (US East)
Branch:               main
Root Directory:       backend
Build Command:        pip install -r requirements-minimal.txt
Start Command:        uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 4.4 Select Free Plan
**Scroll down** and choose **"Free"** ✓

### 4.5 Click Deploy
- Wait 2-3 minutes
- Watch the build logs scroll by
- When done, you'll see:
```
=== Backend Deployment Complete ===
Service: deepfake-detector-api
URL: https://deepfake-detector-api.onrender.com
```

**Save this URL!** You need it for the frontend.

### 4.6 Test Backend (Optional)
```bash
curl https://deepfake-detector-api.onrender.com/health
```

Should return:
```json
{"status": "healthy", "mode": "custom_demo", ...}
```

---

## STEP 5: Deploy Frontend on Vercel

### 5.1 Go to Vercel
1. **[vercel.com](https://vercel.com)**
2. Click **"Sign Up"** → **"Continue with GitHub"**
3. Authorize Vercel

### 5.2 Import Project
1. Click **"Add New..."** (top right)
2. Select **"Project"**
3. Search and select: `finalYrProject`
4. Click **"Import"**

### 5.3 Configure Project
**Fill in exactly:**

```
Project Name:         deepfake-detector
Framework Preset:     Next.js
Root Directory:       frontend
```

### 5.4 Add Environment Variable
**CRITICAL!** This connects your frontend to your backend.

Scroll down to **"Environment Variables"**
Click **"Add"**

Fill in:
```
Name:  NEXT_PUBLIC_API_URL
Value: https://deepfake-detector-api.onrender.com
```

(Use YOUR backend URL from Step 4.5!)

### 5.5 Click Deploy
- Wait 1-2 minutes
- When done, you'll see:
```
✓ Deployment Complete!
https://deepfake-detector.vercel.app
```

**This is your LIVE SITE!** 🎉

---

## STEP 6: Test Everything

### 6.1 Visit Frontend
1. Open: `https://deepfake-detector.vercel.app`
2. Should see the app load (might take 5-10 seconds)
3. See the upload interface

### 6.2 Upload Test Image
1. Drag & drop an image or click "Choose Image"
2. Wait for prediction (might be slow first time - Render is waking up)
3. Should see result like: **"REAL - 85% confidence"**

### 6.3 Troubleshoot if Issues

**Issue: "Network error" or blank page**
- Check if Vercel shows "Ready" (green checkmark)
- Check if Render shows "Live" 
- Wait 30 seconds and refresh

**Issue: "Failed to fetch from API"**
- Check `NEXT_PUBLIC_API_URL` is set in Vercel
- Visit `https://deepfake-detector-api.onrender.com/health` in browser
- Should show JSON response

**Issue: Image doesn't process (takes 30+ seconds)**
- Render free tier is sleeping
- First request takes 10-30 seconds to wake up
- Try again or wait longer

---

## STEP 7: Share Your Live Site!

Your app is now **LIVE on the internet!** 🎉

### Share With:
```
Frontend URL: https://deepfake-detector.vercel.app
GitHub Repo:  https://github.com/YOUR_USERNAME/finalYrProject
```

Send to:
- Friends: "I built an AI deepfake detector!"
- Social Media: Share the URL
- Portfolio: Add link to your projects
- GitHub: Star and follow

---

## STEP 8: Make Updates (From Now On)

### Update Code
```bash
# Make changes locally
cd /Users/deepak/Documents/explore/finalYrProject

# Stage changes
git add .

# Commit
git commit -m "Add feature / Fix bug"

# Push to GitHub
git push origin main
```

**Both Render and Vercel auto-deploy in 1-3 minutes!** ✅

No manual steps needed after first deployment.

---

## STEP 9: Monitor Your Deployment

### Render Dashboard
- Check at: https://render.com/dashboard
- Click `deepfake-detector-api`
- See:
  - Build logs
  - Runtime logs
  - Resource usage
  - Restart service if needed

### Vercel Dashboard
- Check at: https://vercel.com/dashboard
- Click `deepfake-detector`
- See:
  - Deployment history
  - Build logs
  - Analytics (visitors, page views)
  - Performance metrics

---

## STEP 10: Optimize Performance

### First Request is Slow?
- Render free tier sleeps for 15 min inactivity
- First request wakes it up (10-30 seconds)
- Subsequent requests are fast (2-3 seconds)

**To keep it awake, visit every 10 minutes**
(Or upgrade to paid tier)

### Make It Faster (Optional)
Edit `backend/api/main.py`:
```python
# Add this at the top
import threading
import time

def ping_self():
    import requests
    while True:
        time.sleep(600)
        try:
            requests.get("https://deepfake-detector-api.onrender.com/health")
        except:
            pass

# Start thread
threading.Thread(target=ping_self, daemon=True).start()
```

Then: `git push` and Render will auto-update!

---

## FINAL CHECKLIST

- [ ] Created GitHub account
- [ ] Created GitHub repository (finalYrProject)
- [ ] Pushed code to GitHub (all 150+ files)
- [ ] Deployed backend on Render (/health works)
- [ ] Deployed frontend on Vercel
- [ ] Set NEXT_PUBLIC_API_URL in Vercel
- [ ] Frontend loads in browser
- [ ] Can upload image
- [ ] Get prediction result
- [ ] Results display correctly
- [ ] Saved Render URL
- [ ] Saved Vercel URL
- [ ] Can make quick fixes (git push auto-deploys)

---

## SUCCESS! 🎉

Your deepfake detector is now:
✅ Trained with AI
✅ Deployed to the cloud
✅ Live on the internet
✅ Accessible to anyone
✅ Automatically updating

**Share your URL:**
```
https://deepfake-detector.vercel.app
```

You did it! 🚀
