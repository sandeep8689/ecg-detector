"""
ECG Heart Anomaly Detector - FastAPI Backend
=============================================
Model: EfficientNet-B4 fine-tuned on ECG Image Dataset (Kaggle)
Accuracy: ~96.4% on validation set
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

from routes.analyze import router as analyze_router
from routes.report import router as report_router

app = FastAPI(
    title="ECG Heart Anomaly Detector API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/api")
app.include_router(report_router, prefix="/api")

# Serve generated reports
os.makedirs("generated_reports", exist_ok=True)
app.mount("/reports", StaticFiles(directory="generated_reports"), name="reports")

@app.get("/")
def root():
    return {"status": "ECG Detector API running ✅", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
