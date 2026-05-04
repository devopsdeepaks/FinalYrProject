#!/bin/bash

# 🚀 QUICKSTART: Deploy Deepfake Detector in 5 Steps
# This script guides you through training, testing, and deploying

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   🎬 DEEPFAKE DETECTOR - DEPLOYMENT QUICKSTART                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Check Prerequisites
# ─────────────────────────────────────────────────────────────────────

echo "📋 STEP 1: Checking Prerequisites..."
echo ""

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ $1 is not installed. Please install it first."
        exit 1
    fi
    echo "✓ $1 found"
}

check_command "python3"
check_command "git"
echo ""

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Verify Training Data
# ─────────────────────────────────────────────────────────────────────

echo "📊 STEP 2: Verifying Training Data..."
echo ""

FAKE_COUNT=$(ls -1 backend/data/frames/fake | wc -l)
REAL_COUNT=$(ls -1 backend/data/frames/real | wc -l)

echo "Fake images: $FAKE_COUNT"
echo "Real images: $REAL_COUNT"

if [ "$FAKE_COUNT" -lt 50 ] || [ "$REAL_COUNT" -lt 50 ]; then
    echo "❌ Need at least 50 fake and 50 real images!"
    echo "Current: $FAKE_COUNT fake, $REAL_COUNT real"
    exit 1
fi

echo "✓ Training data verified (total: $((FAKE_COUNT + REAL_COUNT)) images)"
echo ""

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Train Model
# ─────────────────────────────────────────────────────────────────────

echo "🤖 STEP 3: Training Model..."
echo ""
echo "This will take 10-30 minutes. Please wait..."
echo ""

cd backend
python3 quick_train_demo.py

if [ ! -f "checkpoints/demo_model.pth" ]; then
    echo "❌ Training failed! Model not saved."
    exit 1
fi

MODEL_SIZE=$(ls -lh checkpoints/demo_model.pth | awk '{print $5}')
echo "✓ Model trained! Size: $MODEL_SIZE"
echo ""
cd ..

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Test Backend Locally
# ─────────────────────────────────────────────────────────────────────

echo "🧪 STEP 4: Testing Backend API..."
echo ""
echo "Starting backend server..."
cd backend

# Start server in background
timeout 15 python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
sleep 3

# Test health endpoint
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Backend server is running"
else
    echo "❌ Backend server failed to start"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Test predict endpoint with sample image
if [ -f "../samples/real/001.jpg" ]; then
    RESPONSE=$(curl -s -X POST -F "file=@../samples/real/001.jpg" http://localhost:8000/predict)
    if echo "$RESPONSE" | grep -q "is_fake"; then
        echo "✓ Prediction API working"
    else
        echo "⚠ Prediction returned: $RESPONSE"
    fi
fi

kill $SERVER_PID 2>/dev/null || true
echo "✓ Backend test completed"
echo ""
cd ..

# ─────────────────────────────────────────────────────────────────────
# STEP 5: Setup Git
# ─────────────────────────────────────────────────────────────────────

echo "📦 STEP 5: Setting Up Version Control..."
echo ""

if [ ! -d ".git" ]; then
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git already initialized"
fi

if [ ! -f ".gitignore" ]; then
    echo "❌ .gitignore not found"
    exit 1
fi

echo "✓ .gitignore configured"
echo ""

# ─────────────────────────────────────────────────────────────────────
# STEP 6: Deployment Instructions
# ─────────────────────────────────────────────────────────────────────

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   ✅ LOCAL TESTING COMPLETE!                                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 NEXT STEPS FOR DEPLOYMENT:"
echo ""
echo "1️⃣  Push to GitHub:"
echo "   git add ."
echo "   git commit -m 'Train model with 100 images'"
echo "   git push origin main"
echo ""
echo "2️⃣  Deploy Backend (choose one):"
echo "   • Render: https://render.com"
echo "   • Railway: https://railway.app"
echo ""
echo "3️⃣  Deploy Frontend:"
echo "   • Vercel: https://vercel.app"
echo "   (Set NEXT_PUBLIC_API_URL env var)"
echo ""
echo "4️⃣  Test Live Site:"
echo "   Upload an image → Get instant deepfake detection!"
echo ""
echo "📖 Full guide: See DEPLOYMENT_GUIDE.md"
echo ""
echo "Need help? Check the guide for troubleshooting!"
echo ""
