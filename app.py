from flask import Flask, request, jsonify, render_template_string
import numpy as np
from PIL import Image
import io
import tensorflow.lite as tflite

MODEL_PATH = "waste_classifier.tflite"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 224

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Segregacja Odpadow AI</title>
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
            justify-content: center;
            padding: 20px;
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
        .upload-area:hover { background: #243d24; border-color: #66bb6a; }
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
        }
        .result.show { display: block; background: #1a2e1a; border: 2px solid #4caf50; }
        .result-class { font-size: 1.8rem; font-weight: bold; margin: 8px 0; }
        .result-confidence { color: #aaa; font-size: 1.1rem; }
        .preview-img { max-width: 300px; max-height: 300px; border-radius: 12px; margin-top: 16px; }
        .loading { color: #4caf50; font-size: 1.2rem; margin-top: 16px; display: none; }
        .bar-container { margin-top: 16px; max-width: 300px; margin-left: auto; margin-right: auto; }
        .bar-label { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px; }
        .bar-bg { background: #333; border-radius: 8px; height: 24px; overflow: hidden; margin-bottom: 8px; }
        .bar-fill { height: 100%; border-radius: 8px; background: #4caf50; transition: width 0.5s; }
    </style>
</head>
<body>
    <h1>Segregacja Odpadów</h1>
    <p class="subtitle">Wrzuć zdjecie odpadu - AI powie Ci do ktorego kosza go wyrzucić</p>

    <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
        <div class="upload-icon">📸</div>
        <p>Kliknij lub przeciągnij zdjęcie tutaj</p>
    </div>
    <input type="file" id="fileInput" accept="image/*">

    <div class="loading" id="loading">Analizuję zdjęcie...</div>

    <div class="result" id="result">
        <img class="preview-img" id="preview">
        <div class="result-class" id="resultClass"></div>
        <div class="result-confidence" id="resultConf"></div>
        <div class="bar-container" id="bars"></div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');

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
                    document.getElementById('resultClass').textContent = data['class'];
                    document.getElementById('resultConf').textContent = 'Pewnosc: ' + (data.confidence * 100).toFixed(1) + '%';

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
                })
                .catch(function() {
                    document.getElementById('resultClass').textContent = 'Blad';
                    document.getElementById('resultConf').textContent = 'Sprobuj ponownie';
                    result.className = 'result show';
                });
            loading.style.display = 'none';
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
        return jsonify({"error": "Brak obrazu"}), 400

    try:
        img = Image.open(request.files["image"]).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        predicted_idx = int(np.argmax(output))
        confidence = float(np.max(output))

        return jsonify({
            "class": CLASS_NAMES[predicted_idx],
            "confidence": confidence,
            "all_predictions": {cls: float(prob) for cls, prob in zip(CLASS_NAMES, output[0])}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
