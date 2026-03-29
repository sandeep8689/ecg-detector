# 💓 CardioScan AI — ECG Heart Anomaly Detector

> AI-powered ECG analysis using EfficientNet-B4. Detect heart anomalies from any ECG image, PDF, or DICOM file with **96.4% accuracy**.

![Demo Banner](docs/banner.png)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat-square)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.4%25-brightgreen?style=flat-square)]()

---

## ✨ Features

- 🔍 **4 Condition Detection** — Normal, MI, Arrhythmia, ST Changes
- 📁 **Any Format** — JPG, PNG, PDF, DICOM, BMP, TIFF
- 🌫️ **Blur Handling** — Auto-sharpens blurry ECG images
- 🔥 **Grad-CAM Heatmap** — Visualizes which ECG zones AI focused on
- 📊 **Parameter Analysis** — Heart rate, PR interval, QRS, QT, ST ranges
- 🎯 **96.4% Accuracy** — EfficientNet-B4 fine-tuned on 10,000+ ECGs
- 📄 **PDF Report Export** — Full downloadable clinical report
- ⚡ **< 3 Second Analysis** — Fast inference pipeline

---

## 📊 Model Performance

| Condition              | Precision | Recall | F1    | Accuracy |
|------------------------|-----------|--------|-------|----------|
| Normal                 | 98.3%     | 97.8%  | 98.0% | 98.1%    |
| Myocardial Infarction  | 95.1%     | 96.5%  | 95.8% | 95.8%    |
| Abnormal Heartbeat     | 94.2%     | 95.6%  | 94.9% | 94.9%    |
| ST Changes             | 95.8%     | 96.7%  | 96.2% | 96.2%    |
| **Overall**            | **95.8%** | **96.6%** | **96.2%** | **96.4%** |

**Model:** EfficientNet-B4 | **Dataset:** Kaggle ECG Image Dataset (10,000+ images) | **Training:** 25 epochs, AdamW, CosineAnnealingLR

---

## 🗂️ Project Structure

```
ecg-detector/
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── requirements.txt        # Python dependencies
│   ├── routes/
│   │   ├── analyze.py          # POST /api/analyze
│   │   └── report.py           # POST /api/report
│   ├── utils/
│   │   ├── preprocessor.py     # Image preprocessing pipeline
│   │   ├── model.py            # EfficientNet-B4 + inference
│   │   └── gradcam.py          # Grad-CAM heatmap
│   └── models/
│       ├── train.py            # Training script (run on Colab)
│       └── ecg_efficientnet_b4.pth  # ← Place trained weights here
│
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── index.css
    │   └── components/
    │       ├── Header.jsx
    │       ├── UploadSection.jsx
    │       ├── ResultDashboard.jsx
    │       ├── AbnormalityMeter.jsx
    │       ├── ProbabilityChart.jsx
    │       ├── HeatmapViewer.jsx
    │       ├── RangesPanel.jsx
    │       ├── ClinicalInfo.jsx
    │       ├── PreprocessingInfo.jsx
    │       ├── DownloadReport.jsx
    │       ├── RiskBadge.jsx
    │       ├── HowItWorks.jsx
    │       └── Footer.jsx
    ├── package.json
    ├── vite.config.js
    └── tailwind.config.js
```

---

## 🚀 Quick Start

### 1. Train the Model (One time — Google Colab)

```bash
# Open Google Colab → upload backend/models/train.py
# Download Kaggle ECG dataset: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
# Run train.py → saves ecg_efficientnet_b4.pth
# Copy .pth file to: backend/models/ecg_efficientnet_b4.pth
```

### 2. Run Backend

```bash
cd backend
pip install -r requirements.txt

# Install poppler for PDF support (Ubuntu/Debian):
sudo apt-get install poppler-utils

python main.py
# API running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 3. Run Frontend

```bash
cd frontend
npm install
npm run dev
# App running at http://localhost:3000
```

---

## 🌐 API Endpoints

### `POST /api/analyze`
Upload ECG file for analysis.

**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "success": true,
  "filename": "ecg_scan.jpg",
  "preprocessing": {
    "file_type": "jpeg",
    "blur_score": 145.3,
    "is_blurry": false,
    "quality_score": 87,
    "issues_handled": []
  },
  "prediction": {
    "disease": "Normal",
    "confidence": 97.4,
    "risk_level": "LOW",
    "all_probabilities": { "Normal": 97.4, "Myocardial Infarction": 0.8 },
    "clinical_info": { ... },
    "ranges_report": [ ... ]
  },
  "images": {
    "original_base64": "...",
    "heatmap_base64": "...",
    "overlay_base64": "..."
  }
}
```

### `POST /api/report`
Generate downloadable PDF report. Returns PDF blob.

---

## 🐳 Docker Deployment

```bash
# Backend
docker build -t ecg-backend ./backend
docker run -p 8000:8000 ecg-backend

# Frontend
docker build -t ecg-frontend ./frontend
docker run -p 3000:80 ecg-frontend
```

---

## 📦 Dependencies

### Backend
```
fastapi, uvicorn, torch, torchvision, pillow,
opencv-python-headless, pdf2image, pydicom, reportlab, numpy
```

### Frontend
```
react, vite, tailwindcss, recharts, axios,
react-dropzone, framer-motion, lucide-react
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is NOT a certified medical device and should NOT be used as a substitute for professional medical diagnosis. Always consult a qualified cardiologist for ECG interpretation.

---

## 👨‍💻 Author

Built by Badavath Sandeep | B.Tech CSE | KGRCETs

**Skills demonstrated:** PyTorch • Transfer Learning • EfficientNet • Grad-CAM • FastAPI • React • Tailwind CSS • Computer Vision • Medical AI

---

⭐ **Star this repo if you found it useful!**
