
from flask import Flask, request, jsonify
import tensorflow.lite as tflite
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
model = load_model("waste_classifier.h5")
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Brak obrazu"}), 400

    img = Image.open(request.files["image"]).resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    pred = model.predict(img_array)
    result = {
        "class": CLASS_NAMES[np.argmax(pred)],
        "confidence": float(np.max(pred)),
        "all_predictions": {c: float(p) for c, p in zip(CLASS_NAMES, pred[0])}
    }
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
