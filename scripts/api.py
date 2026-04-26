import logging
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from countries import load_classes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

THRESHOLD = float(os.environ.get("PREDICT_THRESHOLD", 50.0))
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 Mo

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

log.info("Chargement du modèle...")
model = tf.keras.models.load_model("models/best_pedigree_model.h5")
CLASSES = load_classes()
log.info("Modèle prêt (%d classes).", len(CLASSES))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": len(CLASSES)})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Champ 'file' manquant."}), 400

    file = request.files["file"]
    if file.mimetype not in ALLOWED_MIME:
        return jsonify({"error": f"Type MIME non supporté : {file.mimetype}"}), 415

    try:
        img = Image.open(file.stream).convert("RGB").resize((224, 224))
    except UnidentifiedImageError:
        return jsonify({"error": "Fichier image invalide ou corrompu."}), 422
    except Exception as exc:
        log.exception("Erreur ouverture image")
        return jsonify({"error": str(exc)}), 500

    img_array = (np.array(img) / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        preds = model.predict(img_array, verbose=0)[0]
    except Exception as exc:
        log.exception("Erreur inférence")
        return jsonify({"error": "Erreur lors de la prédiction."}), 500

    top_idx = np.argsort(preds)[::-1][:3]
    results = [
        {"country": CLASSES[i], "score": round(float(preds[i]) * 100, 2)}
        for i in top_idx
    ]
    status = "success" if results[0]["score"] >= THRESHOLD else "uncertain"

    log.info("predict → %s (%.1f%%)", results[0]["country"], results[0]["score"])
    return jsonify({"status": status, "predictions": results})


if __name__ == "__main__":
    # Dev uniquement — en production : gunicorn -w 2 scripts.api:app
    app.run(host="0.0.0.0", port=5000)