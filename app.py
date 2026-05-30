from flask import Flask, request, jsonify, render_template_string
import numpy as np
from PIL import Image
import io
import os
import requests as http_requests
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

# Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://psoynddgcmfxmbzigmwf.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_Fpy3QYbMq30aD-aCtMzGPg_Vou1y5BE")
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

def supabase_insert(data):
    try:
        r = http_requests.post(f"{SUPABASE_URL}/rest/v1/predictions", json=data, headers=HEADERS, timeout=5)
        if r.status_code < 300:
            result = r.json()
            if result and len(result) > 0:
                return result[0].get("id")
        return None
    except:
        return None

def supabase_update(record_id, data):
    try:
        h = dict(HEADERS)
        h["Prefer"] = "return=minimal"
        http_requests.patch(f"{SUPABASE_URL}/rest/v1/predictions?id=eq.{record_id}", json=data, headers=h, timeout=5)
    except:
        pass

def supabase_read():
    try:
        r = http_requests.get(f"{SUPABASE_URL}/rest/v1/predictions?order=timestamp.desc&limit=100", headers=HEADERS, timeout=5)
        if r.status_code < 300:
            return r.json()
        return []
    except:
        return []

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
        .preview-img { max-width: 280px; max-height: 280px; border-radius: 12px; margin-bottom: 16px; }
        .result-class { font-size: 2rem; font-weight: bold; color: #4caf50; margin: 8px 0; text-transform: uppercase; }
        .result-confidence { color: #aaa; font-size: 1.1rem; margin-bottom: 16px; }

        .bin-info {
            background: #243d24;
            border-radius: 12px;
            padding: 16px;
            margin: 12px 0;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .bin-icon { font-size: 2rem; margin-bottom: 8px; }

        /* Feedback */
        .feedback-section { margin-top: 20px; }
        .feedback-question { color: #aaa; font-size: 0.95rem; margin-bottom: 12px; }
        .feedback-btns { display: flex; gap: 12px; justify-content: center; }
        .btn-correct, .btn-wrong {
            padding: 10px 24px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            font-weight: bold;
            transition: all 0.2s;
        }
        .btn-correct { background: #2e7d32; color: white; }
        .btn-correct:hover { background: #388e3c; }
        .btn-wrong { background: #c62828; color: white; }
        .btn-wrong:hover { background: #d32f2f; }

        .correction-section {
            display: none;
            margin-top: 16px;
        }
        .correction-section.show { display: block; }
        .correction-label { color: #aaa; font-size: 0.9rem; margin-bottom: 10px; }
        .correction-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
            gap: 8px;
        }
        .correction-btn {
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #4caf5050;
            background: #1a2e1a;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        .correction-btn:hover { background: #4caf5030; border-color: #4caf50; }

        .feedback-thanks {
            display: none;
            color: #4caf50;
            font-size: 0.95rem;
            margin-top: 12px;
            padding: 12px;
            background: #243d24;
            border-radius: 10px;
        }
        .feedback-thanks.show { display: block; }

        /* Details */
        .details-toggle {
            background: none;
            border: 1px solid #4caf5050;
            color: #4caf50;
            padding: 8px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85rem;
            margin-top: 16px;
            transition: all 0.3s;
        }
        .details-toggle:hover { background: #4caf5020; }

        .details { display: none; margin-top: 16px; text-align: left; width: 100%; }
        .details.show { display: block; }
        .details-title { font-size: 0.85rem; color: #888; margin-bottom: 8px; }
        .bar-container { max-width: 100%; }
        .bar-label { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px; color: #ccc; }
        .bar-bg { background: #333; border-radius: 6px; height: 20px; overflow: hidden; margin-bottom: 8px; }
        .bar-fill { height: 100%; border-radius: 6px; background: #4caf50; transition: width 0.5s; }
        .model-note { font-size: 0.8rem; color: #666; margin-top: 12px; padding-top: 12px; border-top: 1px solid #333; }

        .loading { color: #4caf50; font-size: 1.2rem; margin-top: 16px; display: none; }
        .bottom-links { margin-top: 32px; display: flex; gap: 24px; }
        .bottom-link { color: #4caf50; text-decoration: none; font-size: 0.9rem; }
        .bottom-link:hover { text-decoration: underline; }
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

    <div class="loading" id="loading">Analyzing image (removing background)...</div>

    <div class="result" id="result">
        <img class="preview-img" id="preview">
        <div class="result-class" id="resultClass"></div>
        <div class="result-confidence" id="resultConf"></div>
        <div class="bin-info" id="binInfo"></div>

        <div class="feedback-section" id="feedbackSection">
            <div class="feedback-question">Is this classification correct?</div>
            <div class="feedback-btns">
                <button class="btn-correct" onclick="sendFeedback(true)">✅ Correct</button>
                <button class="btn-wrong" onclick="sendFeedback(false)">❌ Wrong</button>
            </div>
        </div>

        <div class="correction-section" id="correctionSection">
            <div class="correction-label">What is the correct category?</div>
            <div class="correction-grid">
                <button class="correction-btn" onclick="sendCorrection('cardboard')">📦 Cardboard</button>
                <button class="correction-btn" onclick="sendCorrection('glass')">🫙 Glass</button>
                <button class="correction-btn" onclick="sendCorrection('metal')">🥫 Metal</button>
                <button class="correction-btn" onclick="sendCorrection('paper')">📄 Paper</button>
                <button class="correction-btn" onclick="sendCorrection('plastic')">🧴 Plastic</button>
                <button class="correction-btn" onclick="sendCorrection('trash')">🗑️ Trash</button>
                <button class="correction-btn" onclick="sendCorrection('organic')">🥬 Organic</button>
            </div>
        </div>

        <div class="feedback-thanks" id="feedbackThanks"></div>

        <button class="details-toggle" onclick="toggleDetails()">Show model details</button>

        <div class="details" id="details">
            <div class="details-title">Prediction breakdown (6-class model)</div>
            <div class="bar-container" id="bars"></div>
            <div class="model-note" id="modelNote"></div>
        </div>
    </div>

    <div class="bottom-links">
        <a class="bottom-link" href="/stats">📊 Statistics</a>
    </div>

    <script>
        var BIN_INFO = {
            'cardboard': { icon: '📦', text: 'Dispose in the paper/cardboard recycling bin. Flatten boxes before recycling.' },
            'glass': { icon: '🫙', text: 'Dispose in the glass recycling bin. Remove caps and rinse before recycling.' },
            'metal': { icon: '🥫', text: 'Dispose in the metal recycling bin. Rinse cans before recycling.' },
            'paper': { icon: '📄', text: 'Dispose in the paper recycling bin. Keep dry and clean.' },
            'plastic': { icon: '🧴', text: 'Dispose in the plastic recycling bin. Check the recycling number on the item.' },
            'trash': { icon: '🗑️', text: 'Dispose in the general waste bin. This item cannot be recycled.' },
            'organic': { icon: '🥬', text: 'Dispose in the organic/compost bin. Suitable for composting.' }
        };
        var UNSURE_THRESHOLD = 0.3;
        var currentPredictionId = null;

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

        function toggleDetails() {
            var d = document.getElementById('details');
            var btn = document.querySelector('.details-toggle');
            if (d.classList.contains('show')) {
                d.classList.remove('show');
                btn.textContent = 'Show model details';
            } else {
                d.classList.add('show');
                btn.textContent = 'Hide model details';
            }
        }

        function handleFile(file) {
            var reader = new FileReader();
            reader.onload = function(e) { document.getElementById('preview').src = e.target.result; };
            reader.readAsDataURL(file);

            loading.style.display = 'block';
            result.classList.remove('show');
            document.getElementById('details').classList.remove('show');
            document.querySelector('.details-toggle').textContent = 'Show model details';
            document.getElementById('feedbackSection').style.display = 'block';
            document.getElementById('correctionSection').classList.remove('show');
            document.getElementById('feedbackThanks').classList.remove('show');

            var formData = new FormData();
            formData.append('image', file);

            fetch('/predict', { method: 'POST', body: formData })
                .then(function(res) { return res.json(); })
                .then(function(data) {
                    currentPredictionId = data.prediction_id;
                    var cls = data.final_class;
                    var info = BIN_INFO[cls] || BIN_INFO['trash'];

                    if (data.confidence < UNSURE_THRESHOLD) {
                    document.getElementById('resultClass').textContent = 'NOT SURE';
                    document.getElementById('resultConf').textContent = 'Confidence too low: ' + (data.confidence * 100).toFixed(1) + '%';
                    document.getElementById('binInfo').innerHTML = '<div class="bin-icon">🤔</div>Try a clearer photo with the object centered and good lighting.';
                        } else
                        {
                    document.getElementById('resultClass').textContent = data.final_class;
                    document.getElementById('resultConf').textContent = 'Confidence: ' + (data.confidence * 100).toFixed(1) + '%';
                    document.getElementById('binInfo').innerHTML = '<div class="bin-icon">' + info.icon + '</div>' + info.text;
                        }
                    document.getElementById('resultConf').textContent = 'Confidence: ' + (data.confidence * 100).toFixed(1) + '%';
                    document.getElementById('binInfo').innerHTML = '<div class="bin-icon">' + info.icon + '</div>' + info.text;

                    var bars = document.getElementById('bars');
                    bars.innerHTML = '';
                    var preds = data.all_predictions;
                    for (var c in preds) {
                        var prob = preds[c];
                        bars.innerHTML +=
                            '<div class="bar-label"><span>' + c + '</span><span>' + (prob*100).toFixed(1) + '%</span></div>' +
                            '<div class="bar-bg"><div class="bar-fill" style="width:' + (prob*100) + '%"></div></div>';
                    }

                    var noteEl = document.getElementById('modelNote');
                    if (data.note) {
                        noteEl.textContent = data.note;
                        noteEl.style.display = 'block';
                    } else {
                        noteEl.style.display = 'none';
                    }

                    result.className = 'result show';
                    loading.style.display = 'none';
                })
                .catch(function() {
                    document.getElementById('resultClass').textContent = 'Error';
                    document.getElementById('resultConf').textContent = 'Please try again';
                    document.getElementById('binInfo').innerHTML = '';
                    result.className = 'result show';
                    loading.style.display = 'none';
                });
        }

        function sendFeedback(correct) {
            if (correct) {
                fetch('/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prediction_id: currentPredictionId, user_agrees: true })
                });
                document.getElementById('feedbackSection').style.display = 'none';
                var thanks = document.getElementById('feedbackThanks');
                thanks.textContent = '✅ Thank you for confirming!';
                thanks.classList.add('show');
            } else {
                document.getElementById('feedbackSection').style.display = 'none';
                document.getElementById('correctionSection').classList.add('show');
            }
        }

        function sendCorrection(correctClass) {
            fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prediction_id: currentPredictionId,
                    user_agrees: false,
                    user_correction: correctClass
                })
            });
            document.getElementById('correctionSection').classList.remove('show');
            var thanks = document.getElementById('feedbackThanks');
            thanks.textContent = '🔄 Thank you! Your correction (' + correctClass + ') has been saved for model improvement.';
            thanks.classList.add('show');
        }
    </script>
</body>
</html>
"""

STATS_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Statistics - Waste Classification AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #0f1b0f;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 40px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 { font-size: 1.8rem; color: #4caf50; margin-bottom: 8px; }
        .subtitle { color: #888; margin-bottom: 32px; }
        .back-link { color: #4caf50; text-decoration: none; margin-bottom: 24px; }
        .back-link:hover { text-decoration: underline; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            max-width: 700px;
            width: 100%;
            margin-bottom: 32px;
        }
        .stat-card {
            background: #1a2e1a;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .stat-number { font-size: 2rem; font-weight: bold; color: #4caf50; }
        .stat-label { color: #888; font-size: 0.85rem; margin-top: 4px; }
        .section {
            max-width: 700px;
            width: 100%;
            background: #1a2e1a;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }
        .section-title { font-size: 1.1rem; margin-bottom: 16px; color: #ccc; }
        .chart-bar-row { display: flex; align-items: center; margin-bottom: 10px; }
        .chart-bar-label { width: 100px; font-size: 0.9rem; color: #aaa; }
        .chart-bar-bg { flex: 1; background: #333; border-radius: 6px; height: 28px; overflow: hidden; }
        .chart-bar-fill { height: 100%; border-radius: 6px; background: #4caf50; }
        .chart-bar-count { margin-left: 8px; font-size: 0.85rem; color: #aaa; min-width: 30px; }
        .history-item {
            display: grid;
            grid-template-columns: 120px 120px 80px 1fr;
            gap: 12px;
            align-items: center;
            padding: 12px 8px;
            border-bottom: 1px solid #2a3e2a;
            font-size: 0.9rem;
        }
        .history-item:nth-child(even) { background: rgba(255,255,255,0.03); }
        }
        .history-item:last-child { border-bottom: none; }
        .history-class { color: #4caf50; font-weight: bold; text-transform: uppercase; }
        .history-time { color: #666; }
        .history-conf { color: #aaa; }
        .history-feedback { font-size: 0.8rem; }
        .fb-correct { color: #4caf50; }
        .fb-wrong { color: #ef5350; }
        .no-data { color: #666; text-align: center; padding: 40px; }
    </style>
</head>
<body>
    <a class="back-link" href="/">← Back to classifier</a>
    <h1>📊 Classification Statistics</h1>
    <p class="subtitle">Data from all predictions</p>
    <div id="content"><div class="no-data">Loading...</div></div>

    <script>
        fetch('/api/stats')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                var el = document.getElementById('content');
                if (!data.total || data.total === 0) {
                    el.innerHTML = '<div class="no-data">No predictions yet. Go classify some waste!</div>';
                    return;
                }

                var html = '<div class="stats-grid">';
                html += '<div class="stat-card"><div class="stat-number">' + data.total + '</div><div class="stat-label">Total predictions</div></div>';
                html += '<div class="stat-card"><div class="stat-number">' + (data.avg_confidence * 100).toFixed(1) + '%</div><div class="stat-label">Avg confidence</div></div>';
                html += '<div class="stat-card"><div class="stat-number">' + data.correct_count + '</div><div class="stat-label">Confirmed correct</div></div>';
                html += '<div class="stat-card"><div class="stat-number">' + data.correction_count + '</div><div class="stat-label">User corrections</div></div>';
                html += '</div>';

                html += '<div class="section"><div class="section-title">Distribution by class</div>';
                var maxCount = 0;
                for (var c in data.distribution) { if (data.distribution[c] > maxCount) maxCount = data.distribution[c]; }
                for (var cls in data.distribution) {
                    var count = data.distribution[cls];
                    var pct = maxCount > 0 ? (count / maxCount * 100) : 0;
                    html += '<div class="chart-bar-row">';
                    html += '<div class="chart-bar-label">' + cls + '</div>';
                    html += '<div class="chart-bar-bg"><div class="chart-bar-fill" style="width:' + pct + '%"></div></div>';
                    html += '<div class="chart-bar-count">' + count + '</div>';
                    html += '</div>';
                }
                html += '</div>';

                html += '<div class="section"><div class="section-title">Recent predictions</div>';
                html += '<div class="history-item" style="color:#4caf50; font-weight:bold; border-bottom:2px solid #4caf50; font-size:0.8rem;">';
                html += '<span>CLASS</span><span>FEEDBACK</span><span>CONF.</span><span>DATE</span></div>';
                for (var i = 0; i < data.recent.length && i < 15; i++) {
                    var p = data.recent[i];
                    var time = new Date(p.timestamp).toLocaleString();
                    var fbHtml = '<span class="history-feedback">—</span>';
                    if (p.user_agrees === true) fbHtml = '<span class="history-feedback fb-correct">✅</span>';
                    else if (p.user_agrees === false) fbHtml = '<span class="history-feedback fb-wrong">❌ → ' + (p.user_correction || '?') + '</span>';
                    html += '<div class="history-item">';
                    html += '<span class="history-class">' + p.final_class + '</span>';
                    html += fbHtml;
                    html += '<span class="history-conf">' + (p.confidence * 100).toFixed(1) + '%</span>';
                    html += '<span class="history-time">' + time + '</span>';
                    html += '</div>';
                }
                html += '</div>';

                el.innerHTML = html;
            });
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

        output_6 = run_6class(img_array)
        pred_6_idx = int(np.argmax(output_6))
        pred_6_class = CLASS_NAMES_6[pred_6_idx]
        pred_6_conf = float(np.max(output_6))
        all_preds = {cls: float(prob) for cls, prob in zip(CLASS_NAMES_6, output_6[0])}

        final_class = pred_6_class
        confidence = pred_6_conf
        note = ""
        model_2class_result = None

        if pred_6_class == "trash":
            output_2 = run_2class(img_array)
            pred_2_idx = int(np.argmax(output_2))
            pred_2_class = CLASS_NAMES_2[pred_2_idx]
            pred_2_conf = float(np.max(output_2))
            model_2class_result = pred_2_class

            if pred_2_class == "Organic":
                final_class = "organic"
                confidence = pred_2_conf
                note = "6-class model detected trash, but 2-class model identified it as organic"
            else:
                final_class = "trash"
                confidence = pred_6_conf
                note = "Confirmed as trash by both models"

        prediction_id = supabase_insert({
            "final_class": final_class,
            "confidence": confidence,
            "model_6class_result": pred_6_class,
            "model_2class_result": model_2class_result,
            "note": note
        })

        return jsonify({
            "final_class": final_class,
            "confidence": confidence,
            "note": note,
            "all_predictions": all_preds,
            "prediction_id": prediction_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    pid = data.get("prediction_id")
    if not pid:
        return jsonify({"error": "No prediction_id"}), 400

    update = {"user_agrees": data.get("user_agrees")}
    if data.get("user_correction"):
        update["user_correction"] = data["user_correction"]

    supabase_update(pid, update)
    return jsonify({"status": "ok"})

@app.route("/stats")
def stats():
    return render_template_string(STATS_PAGE)

@app.route("/api/stats")
def api_stats():
    rows = supabase_read()
    if not rows:
        return jsonify({"total": 0, "distribution": {}, "avg_confidence": 0, "correct_count": 0, "correction_count": 0, "recent": []})

    total = len(rows)
    distribution = {}
    conf_sum = 0
    correct = 0
    corrections = 0

    for r in rows:
        cls = r.get("final_class", "unknown")
        distribution[cls] = distribution.get(cls, 0) + 1
        conf_sum += r.get("confidence", 0)
        if r.get("user_agrees") is True:
            correct += 1
        elif r.get("user_agrees") is False:
            corrections += 1

    return jsonify({
        "total": total,
        "distribution": distribution,
        "avg_confidence": conf_sum / total if total > 0 else 0,
        "correct_count": correct,
        "correction_count": corrections,
        "recent": rows[:15]
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models": ["6class", "2class"], "features": ["rembg", "feedback", "supabase"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
