"""
FastAPI application entry point for deepfake detection.
"""

import sys
from pathlib import Path

# Ensure backend root is importable from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import router

app = FastAPI(
    title="Deepfake Detection API",
    description=(
        "REST API for detecting deepfakes using a hybrid "
        "EfficientNetB4 + Attention + ViT model."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "Deepfake Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/predict"],
    }


if __name__ == "__main__":
    import config
    uvicorn.run(
        "api.main:app",
        host=config.API_CONFIG["host"],
        port=config.API_CONFIG["port"],
        reload=False,
    )
