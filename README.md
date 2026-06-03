# Chest X-ray Pneumonia Detection with Deep Learning

This project focuses on detecting **pneumonia** from chest X-ray images using a deep learning model based on **DenseNet121** with transfer learning and fine-tuning.

The system includes:
- A trained deep learning model
- A **FastAPI-based web application**
- A **custom frontend (HTML/CSS)**
- An **evaluation dashboard**
- **Grad-CAM explainability**

---

## Features

- Upload chest X-ray images (PNG / JPG / JPEG)
- Predict pneumonia probability
- Two decision modes:
  - **High Recall (threshold = 0.30)** → screening mode
  - **Balanced (threshold = 0.50)** → general usage
- Grad-CAM heatmap visualization
- Interactive evaluation dashboard:
  - ROC Curve
  - Confusion matrices
  - Threshold comparison metrics
- Medical disclaimer for safe usage

---

## Project Structure

```text
zature_tespiti/
│
├── app.py                 # FastAPI backend
├── model.py               # Model architecture
├── data.py                # Data generators
├── train.py               # Training script
├── evaluate.py            # Evaluation + metrics export
├── best_model.keras       # Trained model
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html         
│   └── results.html       
│
├── static/
│   └── style.css          
│
├── outputs/
│   ├── metrics.json
│   ├── cm_thr_0.30.png
│   ├── cm_thr_0.50.png
│   └── roc_curve.png
│
└── chest_xray/            # Dataset (not included in repo)
```

## Dataset

The model is trained on the Chest X-ray Pneumonia Dataset:

🔗 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Structure:

chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/

All images are resized to 224×224 and preprocessed using DenseNet preprocess_input.

## Model Architecture

Backbone: DenseNet121 (ImageNet pretrained)
Global Average Pooling
Dense (128, ReLU)
Dropout (0.5)
Output: Sigmoid (binary classification)

##  Training Strategy

Train classifier head (frozen backbone)
Fine-tune last convolutional layers

### Training Details

Loss: Binary Cross-Entropy
Optimizer: Adam
Class imbalance handled via class weighting

### Data augmentation:
Rotation
Brightness
Horizontal flip

### Callbacks

EarlyStopping
ReduceLROnPlateau
ModelCheckpoint

## Evaluation Results

ROC-AUC (threshold independent)

ROC-AUC: 0.951

### Threshold Trade-off Analysis
Mode	Threshold	Accuracy	Precision	Recall	F1
High Recall	0.30	0.816	0.785	0.972	0.868
Balanced	0.50	0.870	0.858	0.951	0.902

### Interpretation
Threshold = 0.30 (High Recall Mode)
Maximizes sensitivity and minimizes missed pneumonia cases (FN↓), but increases false positives.
Threshold = 0.50 (Balanced Mode)
Reduces false positives (FP↓) and improves precision, at the cost of slightly more missed cases.

This demonstrates how decision thresholds can be adjusted depending on clinical priorities (screening vs diagnostic usage).

## Evaluation Dashboard

The application includes a dedicated results dashboard page:

Displays:
ROC Curve
Confusion Matrix (0.30)
Confusion Matrix (0.50)
Metrics table (Accuracy, Precision, Recall, F1)
Automatic trade-off explanation

## Web Application (FastAPI)
Run locally
pip install -r requirements.txt
uvicorn app:app --reload

Open in browser:

http://127.0.0.1:8000

## Grad-CAM Explainability

Grad-CAM highlights the regions contributing most to the prediction.

Helps interpret model decisions
Visualizes focus areas on X-ray
Overlaid directly on the image

## Model

A pre-trained model (best_model.keras) is included for inference.

Training is optional.

## Disclaimer

This application is for educational and research purposes only.

It is not intended for clinical use or medical diagnosis.