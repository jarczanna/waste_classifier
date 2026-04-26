from flask import Flask, request, jsonify, render_template_string
import numpy as np
from PIL import Image
import io
import json
import os
from datetime import datetime
import tensorflow.lite as tflite

IMG_SIZE = 224

# 6-class model
CLASS_NAMES_6 = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
interpreter_6 = tflite.Interpreter(model_path="waste_classifier_6_classes.tflite")
interpreter_6.allocate_tensors()
input_6 = interpreter_6.get_input_details()
output_6 = interpreter_6.get_output_details()

# 2-class model
CLASS_NAMES_2 = ["Organic", "Recyclable"]
interpreter_2 = tflite.Interpreter(model_path="waste_classifier_2_classes.tflite")
interpreter_2.allocate_tensors()
input_2 = interpreter_2.get_input_details()
output_2 = interpreter_2.get_output_details()

LOG_FILE = "predictions_log.json"

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []

def save_log(entry):
    log = load_log()
    log.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f)

def prepare_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def run_6class(img_array):
    interpreter_6.set_tensor(input_6[0]["index"], img_array)
    interpreter_6.invoke()
    return interpreter_6.get_tensor(output_6[0]["index"])

def run_2class(img_array):
    interpreter_2.set_tensor(input_2[0]["index"], img_array)
    interpreter_2.invoke()
    return interpreter_2.get_tensor(output_2[0]["index"])

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Waste Classification AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #0f1b0f;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }
        h1 { font-size: 2rem; color: #4caf50; margin-bottom: 8px; }
        .subtitle { color: #888; margin-bottom: 32px; font-size: 0.95rem; }
        .upload-area {
            border: 2px dashed #4caf50;
            border-radius: 16px;
            padding: 48px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            max-width: 500px;
            width: 100%;
            background: #1a2e1a;
        }
        .upload-area:hover { background: #243d24; }
        .upload-icon { font-size: 3rem; margin-bottom: 16px; }
        input[type="file"] { display: none; }
        .result {
            margin-top: 24px;
            padding: 24px;
            border-radius: 16px;
            text-align: center;
            max-width: 500px;
            width: 100%;
            display: none;
            background: #1a2e1a;
            border: 2px solid #4caf50;
        }
        .result.show { display: block; }
        .result-class { font-size: 1.8rem; font-weight: bold; margin: 8px 0; color: #4caf50; }
        .result-note { color: #aaa; font-size: 0.85rem; margin-top: 4px; font-style: italic; }
        .result-confidence { color: #aaa; font-size: 1.1rem; }
        .preview-img { max-width: 300px; max-height: 300px; border-radius: 12px; margin-top: 16px; }
        .loading { color: #4caf50; font-size: 1.2rem; margin-top: 16px; display: none; }
        .bar-container { margin-top: 16px; max-width: 300px; margin-left: auto; margin-right: auto; }
        .bar-label { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px; }
        .bar-bg { background: #333; border-radius: 8px; height: 24px; overflow: hidden; margin-bottom: 8px; }
        .bar-fill { height: 100%; border-radius: 8px; background: #4caf50; transition: width 0.5s; }
        .stats-link { margin-top: 32px; color: #4caf50; text-decoration: none; font-size: 0.9rem; }
        .stats-link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Waste Classification AI</h1>
    <p class="subtitle">Upload a photo of waste and AI will tell you which bin it belongs to</p>

    <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
        <div class="upload-icon">📸</div>
        <p>Click or drag an image here</p>
    </div>
    <input type="file" id="fileInput" accept="image/*">

    <div class="loading" id="loading">Analyzing image...</div>

    <div class="result" id="result">
        <img class="preview-img" id="preview">
        <div class="result-class" id="resultClass"></div>
        <div class="result-note" id="resultNote"></div>
        <div class="result-confidence" id="resultConf"></div>
        <div class="bar-container" id="bars"></div>
    </div>

    <a class="stats-link" href="/stats">📊 View statistics</a>

    <script>
        var uploadArea = document.getElementById('uploadArea');
        var fileInput = document.getElementById('fileInput');
        var loading = document.getElementById('loading');
        var result = document.getElementById('result');

        uploadArea.addEventListener('dragover', function(e) { e.preventDefault(); });
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', function(e) { if (e.target.files.length) handleFile(e.target.files[0]); });

        function handleFile(file) {
            var reader = new FileReader();
            reader.onload = function(e) { document.getElementById('preview').src = e.target.result; };
            reader.readAsDataURL(file);

            loading.style.display = 'block';
            result.classList.remove('show');

            var formData = new FormData();
            formData.append('image', file);

            fetch('/predict', { method: 'POST', body: formData })
                .then(function(res) { return res.json(); })
                .then(function(data) {
                    document.getElementById('resultClass').textContent = data.final_class;
                    document.getElementById('resultConf').textContent = 'Confidence: ' + (data.confidence * 100).toFixed(1) + '%';
                    document.getElementById('resultNote').textContent = data.note || '';

                    var bars = document.getElementById('bars');
                    bars.innerHTML = '';
                    var preds = data.all_predictions;
                    for (var cls in preds) {
                        var prob = preds[cls];
                        bars.innerHTML +=
                            '<div class="bar-label"><span>' + cls + '</span><span>' + (prob*100).toFixed(1) + '%</span></div>' +
                            '<div class="bar-bg"><div class="bar-fill" style="width:' + (prob*100) + '%"></div></div>';
                    }
                    result.className = 'result show';
                    loading.style.display = 'none';
                })
                .catch(function() {
                    document.getElementById('resultClass').textContent = 'Error';
                    document.getElementById('resultConf').textContent = 'Please try again';
                    document.getElementById('resultNote').textContent = '';
                    result.className = 'result show';
                    loading.style.display = 'none';
                });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        img_array = prepare_image(request.files["image"])

        # Step 1: 6-class model
        output_6 = run_6class(img_array)
        pred_6_idx = int(np.argmax(output_6))
        pred_6_class = CLASS_NAMES_6[pred_6_idx]
        pred_6_conf = float(np.max(output_6))
        all_preds = {cls: float(prob) for cls, prob in zip(CLASS_NAMES_6, output_6[0])}

        final_class = pred_6_class
        confidence = pred_6_conf
        note = ""

        # Step 2: if trash, check with 2-class model
        if pred_6_class == "trash":
            output_2 = run_2class(img_array)
            pred_2_idx = int(np.argmax(output_2))
            pred_2_class = CLASS_NAMES_2[pred_2_idx]
            pred_2_conf = float(np.max(output_2))

            if pred_2_class == "Organic":
                final_class = "organic"
                confidence = pred_2_conf
                note = "6-class model detected trash, but 2-class model identified it as organic"
            else:
                final_class = "trash"
                confidence = pred_6_conf
                note = "Confirmed as trash by both models"

        # Log prediction
        entry = {
            "timestamp": datetime.now().isoformat(),
            "final_class": final_class,
            "confidence": confidence,
            "model_6class_result": pred_6_class,
            "note": note
        }
        save_log(entry)

        return jsonify({
            "final_class": final_class,
            "confidence": confidence,
            "note": note,
            "all_predictions": all_preds
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stats")
def stats():
    log = load_log()
    if not log:
        return jsonify({"message": "No data yet - no images have been classified"})

    total = len(log)
    class_counts = {}
    for entry in log:
        cls = entry["final_class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    avg_confidence = sum(e["confidence"] for e in log) / total
    dual_model_cases = sum(1 for e in log if e.get("note", "") != "")

    return jsonify({
        "total_predictions": total,
        "class_distribution": class_counts,
        "average_confidence": round(avg_confidence, 3),
        "dual_model_cases": dual_model_cases,
        "last_10": log[-10:]
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models": ["6class", "2class"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
