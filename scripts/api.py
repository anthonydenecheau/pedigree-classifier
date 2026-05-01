import json
import logging
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

THRESHOLD = float(os.environ.get("PREDICT_THRESHOLD", 50.0))
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 Mo

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Chargement différé : initialisé au premier appel, après le fork Gunicorn
_model = None
_classes = None


def _get_model():
    global _model, _classes
    if _model is None:
        log.info("Chargement du modèle...")
        _model = tf.keras.models.load_model("models/best_pedigree_model.keras")
        with open("models/class_names.json") as f:
            _classes = json.load(f)
        log.info("Modèle prêt (%d classes).", len(_classes))
    return _model, _classes


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    _, classes = _get_model()
    return jsonify({"status": "ok", "classes": len(classes)})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Champ 'file' manquant."}), 400

    file = request.files["file"]
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Extension non supportée : {ext or 'inconnue'}. Utilisez JPG, PNG ou WEBP."}), 415

    try:
        img = Image.open(file.stream).convert("RGB").resize((224, 224))
    except UnidentifiedImageError:
        return jsonify({"error": "Fichier image invalide ou corrompu."}), 422
    except Exception as exc:
        log.exception("Erreur ouverture image")
        return jsonify({"error": str(exc)}), 500

    model, classes = _get_model()

    img_array = np.expand_dims(np.array(img), axis=0).astype("float32")

    try:
        preds = model.predict(img_array, verbose=0)[0]
    except Exception as exc:
        log.exception("Erreur inférence")
        return jsonify({"error": "Erreur lors de la prédiction."}), 500

    top_idx = np.argsort(preds)[::-1][:3]
    results = [
        {"country": classes[i], "score": round(float(preds[i]) * 100, 2)}
        for i in top_idx
    ]
    status = "success" if results[0]["score"] >= THRESHOLD else "uncertain"

    log.info("predict → %s (%.1f%%)", results[0]["country"], results[0]["score"])
    return jsonify({"status": status, "predictions": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
