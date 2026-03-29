"""
ECG Heart Anomaly Detector - FastAPI Backend
=============================================
Model: EfficientNet-B4 fine-tuned on ECG Image Dataset (Kaggle)
Accuracy: ~96.4% on validation set
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

# ✅ Import routers
from routes.analyze import router as analyze_router
from routes.report import router as report_router

# ✅ Initialize FastAPI app
app = FastAPI(
    title="ECG Heart Anomaly Detector API",
    version="1.0.0"
)

# ✅ Enable CORS (for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Register routers
app.include_router(analyze_router, prefix="/api", tags=["Analyze"])
app.include_router(report_router, prefix="/api", tags=["Report"])

# ✅ Ensure reports folder exists
os.makedirs("generated_reports", exist_ok=True)

# ✅ Serve static files (PDF reports)
app.mount("/reports", StaticFiles(directory="generated_reports"), name="reports")

# ✅ Root endpoint
@app.get("/")
def root():
    return {
        "status": "ECG Detector API running ✅",
        "version": "1.0.0"
    }

# ✅ Health check endpoint (for debugging)
@app.get("/health")
def health():
    return {"status": "healthy"}

# ✅ Run locally (NOT used in Render)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    print("🔥 NEW CODE RUNNING")