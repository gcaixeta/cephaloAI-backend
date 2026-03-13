# CephaloAI — Automated Cephalometric Analysis

> Academic research project for automated detection of cephalometric landmarks in lateral skull X-rays using deep learning.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red?logo=pytorch)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)

---

## Overview

CephaloAI automates the cephalometric analysis workflow performed by orthodontists on lateral skull X-rays. Given an image, the system:

1. **Detects 19 anatomical landmarks** using a deep learning model (fusionVGG19 + DilationInceptionModule)
2. **Computes 8 clinical measurements** (ANB, SNB, SNA, ODI, APDI, FHI, FMA, MW) with Angle's classification
3. **Generates an AI-assisted pre-diagnosis** via Google Gemini 2.5 Flash
4. **Returns an annotated image** with landmark overlays alongside structured results

---

## Research Context

This project was developed as part of an academic research initiative on computer-aided cephalometric diagnosis.

> **Institution:** [IFTM - Instituto Federal do Triangulo Mineiro]
> **Advisor:** [Cicero lima costa]
> **Program:** [Analise e Desenvolvimento de Sistemas]
> **Year:** [2025]

The neural network was trained on a dataset of lateral cephalometric radiographs. The model weights (`Best_Model400it.pt`) are not tracked by version control and must be obtained separately.

---

## System Architecture

```
┌─────────────────────────────────┐
│   React Frontend  (port 5173)   │
│   Vite · TypeScript · Tailwind  │
└────────────┬────────────────────┘
             │ HTTP
┌────────────▼────────────────────┐
│   Flask API  (port 5000)        │
│   PyTorch · OpenCV · VGG19-BN   │
└────────────┬────────────────────┘
             │ HTTP
┌────────────▼────────────────────┐
│   Diagnosis Service (port 3001) │
│   Node.js · Express · Gemini    │
└─────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | PyTorch 2.8, VGG19-BN (ImageNet), custom DilationInceptionModule |
| API | Python 3.11, Flask 3.1, OpenCV, scikit-image |
| Frontend | React 19, TypeScript 5.8, Vite 7, Tailwind CSS 4, Radix UI |
| Diagnosis | Node.js, Express, Google Gemini 2.5 Flash (`@google/genai`) |

---

## Model Architecture

The landmark detection model fuses multi-scale features from VGG19-BN with a dilated inception module:

```
Input (800×640 grayscale X-ray)
    ↓
VGG19-BN — feature maps extracted from 4 intermediate layers
    ↓
fusionVGG19 — feature fusion across scales
    ↓
DilationInceptionModule — dilated convolutions + attention
    ↓
19 heatmaps → argmax → (x, y) landmark coordinates
```

**Output:** 19 anatomical landmark coordinates (normalized), mapped back to image space.

---

## Clinical Measurements

| Measurement | Normal Range | Clinical Meaning |
|---|---|---|
| **ANB** | 3.2° – 5.7° | Sagittal jaw relationship (Class I/II/III) |
| **SNA** | 79.4° – 83.2° | Maxillary position relative to cranial base |
| **SNB** | 74.6° – 78.7° | Mandibular position relative to cranial base |
| **ODI** | 68.4° – 80.5° | Overbite depth indicator |
| **APDI** | 77.6° – 85.2° | Anteroposterior dysplasia indicator |
| **FHI** | 0.65 – 0.75 | Facial height index (posterior/anterior ratio) |
| **FMA** | 26.8° – 31.4° | Frankfort-mandibular plane angle |
| **MW** | 2 – 4.5 mm | Maxillary width / molar relationship |

Each measurement is classified into **Class 1** (normal), **Class 2**, **Class 3**, or **Class 4** (MW only).

---

## Getting Started

### Prerequisites

- Python 3.11 ([mise](https://mise.jdx.dev/) recommended)
- Node.js 18+
- NVIDIA GPU with CUDA (recommended; CPU fallback is supported)
- Google API key (for the diagnosis service)
- Model weights: `api/models/Best_Model400it.pt` (not tracked by git — obtain separately)

### Installation & Running

**1. Flask API (port 5000)**

```bash
cd api/
pip install -r requirements.txt
python app.py
```

**2. React Frontend (port 5173)**

```bash
cd frontent/
npm install
npm run dev
```

**3. Diagnosis Service (port 3001)**

```bash
cd diagnosis/
npm install
# Create a .env file with:
# GOOGLE_API_KEY=your_api_key_here
node server.js
```

---

## API Reference

### `POST /processar-imagem`

Upload a lateral X-ray image and receive landmark coordinates, clinical measurements, and the path to the annotated image.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | image | Lateral cephalometric X-ray (JPEG, PNG, BMP) |

**Response:** `application/json`

```json
{
  "coords": [[x0, y0], [x1, y1], "...", [x18, y18]],
  "angles": {
    "ANB": { "value": 4.1, "class": "1" },
    "SNB": { "value": 76.3, "class": "1" },
    "SNA": { "value": 80.4, "class": "1" },
    "ODI": { "value": 74.2, "class": "1" },
    "APDI": { "value": 81.0, "class": "1" },
    "FHI": { "value": 0.70, "class": "1" },
    "FMA": { "value": 28.5, "class": "1" },
    "MW":  { "value": 3.2,  "class": "1" }
  },
  "image_with_overlay_path": "filename.png"
}
```

---

### `GET /download-imagem/<filename>`

Download the annotated image (PNG) with landmark overlays drawn on it.

**Response:** `image/png` (file attachment)

---

## Model Weights

The trained model checkpoint is **not tracked by git** and must be placed manually before running the API:

```
api/
└── models/
    └── Best_Model400it.pt   ← place the checkpoint here
```

The file is loaded at startup in `app.py` via `ImagemService("models/Best_Model400it.pt")`. If the file is missing, the server will fail to start.

---

## License

This project is intended for **academic and research purposes only**.

<!-- Add license here, e.g.: MIT License, or refer to a LICENSE file -->
