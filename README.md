# Waste Classification AI

Web app that classifies waste photos into 7 categories using MobileNetV2 + transfer learning.
Live demo: https://waste-classifier.onrender.com

**Quick Overview**
Upload a waste photo → AI classifies it (cardboard, glass, metal, paper, plastic, trash, organic)
Dual-model system: 6-class model + 2-class model for organic detection
User feedback loop for future model improvement
Stats dashboard at `/stats`

**Tech Stack**
Flask · TensorFlow Lite · MobileNetV2 · Supabase · Render.com · Google Colab
Run Locally
```bash
pip install -r requirements.txt
python app.py
# → http://localhost:8080
```
API
`POST /predict` — classify image (multipart form, field: `image`)
`POST /feedback` — submit correction (`{prediction_id, user_agrees, user_correction}`)
`GET /stats` — statistics dashboard
`GET /api/stats` — raw stats JSON
Project Structure
```
├── app.py                              # API + frontend
├── waste_classifier_6_classes.tflite   # 6-class model
├── waste_classifier_2_classes.tflite   # 2-class model
├── requirements.txt                    # dependencies
├── render.yaml                         # deploy config
```
