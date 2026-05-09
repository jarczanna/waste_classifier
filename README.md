# Waste Classification AI

A web application that classifies waste into categories using deep learning. Upload a photo of waste and the AI will tell you which bin it belongs to.

**Live demo:** [waste-classifier on Render](https://waste-classifier-q817.onrender.com/)

## What it does

The app uses two MobileNetV2 models trained via transfer learning to classify waste images:

- **6-class model** — classifies into: cardboard, glass, metal, paper, plastic, trash
- **2-class model** — classifies into: organic, recyclable

When the 6-class model detects "trash", the image is automatically passed through the 2-class model to check if it's actually organic waste. The user sees only the final result.

## Tech stack

- **Machine Learning:** TensorFlow, MobileNetV2 (transfer learning), TFLite
- **Backend:** Flask, Gunicorn
- **Database:** Supabase (PostgreSQL)
- **Deployment:** Render.com (free tier)
- **Training:** Google Colab

## How the model was trained

1. Dataset downloaded from Kaggle (Garbage Classification — 2000 images per class)
2. Preprocessing: images resized to 224×224, normalized to [0, 1]
3. Augmentation: random rotation, horizontal flip, zoom, shifts
4. Transfer learning with MobileNetV2 (pretrained on ImageNet, top layers frozen)
5. Fine-tuning: last 30 layers unfrozen, trained with low learning rate (1e-5)
6. Evaluation: confusion matrix, precision/recall, classification report
7. Conversion to TFLite for lightweight deployment

## Project structure

```
├── app.py                          # Flask API + frontend
├── waste_classifier_6_classes.tflite   # 6-class TFLite model
├── waste_classifier_2_classes.tflite   # 2-class TFLite model
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render deployment config
└── README.md
```

## API endpoints

- `GET /` — Web interface for uploading and classifying images
- `POST /predict` — Accepts an image, returns classification result as JSON
- `GET /stats` — Dashboard with prediction statistics
- `GET /api/stats` — Raw statistics data as JSON
- `GET /health` — Health check

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

The app will be available at `http://localhost:8080`.

## Environment variables

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anon public key |
| `PORT` | Server port (default: 8080) |
