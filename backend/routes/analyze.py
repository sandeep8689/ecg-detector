"""
/api/analyze — Main ECG Analysis Endpoint
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import traceback

from utils.preprocessor import preprocess_ecg
from utils.model import predict, CLASSES
from utils.gradcam import generate_heatmap_overlay

router = APIRouter()

ALLOWED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
    '.pdf', '.dcm', '.webp'
}
MAX_FILE_SIZE_MB = 50


@router.post("/analyze")
async def analyze_ecg(file: UploadFile = File(...)):
    try:
        import os

        # ── Validate file ─────────────────────────────
        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}"
            )

        file_bytes = await file.read()

        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({size_mb:.1f} MB)"
            )

        if len(file_bytes) < 100:
            raise HTTPException(status_code=400, detail="File is empty")

        # ── Step 1: Preprocess ─────────────────────────
        preprocessing = preprocess_ecg(file_bytes, file.filename)
        processed_image = preprocessing["processed_image"]

        # ── Step 2: Predict ────────────────────────────
        prediction = predict(processed_image)

        # 🔥 FIX: convert numpy types in prediction
        prediction["confidence"] = float(prediction.get("confidence", 0))
        
        if "all_probabilities" in prediction:
            prediction["all_probabilities"] = {
                k: float(v) for k, v in prediction["all_probabilities"].items()
            }

        # ── Step 3: Grad-CAM ───────────────────────────
        predicted_idx = CLASSES.index(prediction["disease"])

        try:
            heatmaps = generate_heatmap_overlay(processed_image, predicted_idx)
        except Exception as e:
            print(f"Heatmap failed: {e}")
            heatmaps = {
                "heatmap_base64": None,
                "overlay_base64": None,
                "original_base64": None
            }

        # ── Step 4: Build Response ─────────────────────
        response = {
            "success": True,
            "filename": file.filename,
            "file_size_mb": float(round(size_mb, 2)),

            "preprocessing": {
                "original_size": [int(x) for x in preprocessing["original_shape"][:2]],
                "file_type": str(preprocessing["file_type"]),
                "blur_score": float(preprocessing["blur_score"]),
                "is_blurry": bool(preprocessing["is_blurry"]),
                "quality_score": float(preprocessing["quality_score"]),
                "issues_handled": list(preprocessing["issues_detected"])
            },

            "prediction": prediction,
            "images": heatmaps,

            "disclaimer": (
                "⚠️ This AI analysis is for informational purposes only. "
                "Not a substitute for medical diagnosis."
            )
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )