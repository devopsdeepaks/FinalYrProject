#!/bin/bash

# Setup script for Deepfake Detection Backend
# This script sets up the development environment

set -e  # Exit on error

echo "🚀 Setting up Deepfake Detection Backend..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create necessary directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p checkpoints
mkdir -p logs
mkdir -p data
mkdir -p results
mkdir -p temp

echo -e "${GREEN}✓ Directories created${NC}"

# Check CUDA availability
echo -e "${BLUE}Checking CUDA availability...${NC}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || echo -e "${RED}PyTorch not installed yet${NC}"

# Download face detection models
echo -e "${BLUE}Setting up face detection models...${NC}"
python3 << EOF
try:
    # MTCNN will download automatically on first use
    from mtcnn import MTCNN
    print("✓ MTCNN available")
except ImportError:
    print("⚠ MTCNN not available (will be installed with requirements)")

try:
    import dlib
    print("✓ Dlib available")
except ImportError:
    print("⚠ Dlib not available (optional)")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__} available")
except ImportError:
    print("⚠ OpenCV not available")
EOF

echo -e "${GREEN}✓ Face detection setup complete${NC}"

# Print summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Setup Complete! 🎉                    ║${NC}"
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Continue implementing Vision Transformer"
echo "  2. Build the ensemble model"
echo "  3. Implement training pipeline"
echo "  4. Download datasets (FaceForensics++, DFDC, COCOFake)"
echo ""
echo "For more information, see README.md"
echo ""
