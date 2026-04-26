from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)
THRESHOLD = 50.0
model = tf.keras.models.load_model('models/best_pedigree_model.h5')
with open('models/classes.json', 'r') as f:
    CLASSES = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB').resize((224, 224))
    img_array = (np.array(img) / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)[0]
    top_idx = np.argsort(preds)[::-1][:3]
    
    results = [{"country": CLASSES[i], "score": round(float(preds[i])*100, 2)} for i in top_idx]
    
    status = "success" if results[0]["score"] >= THRESHOLD else "uncertain"
    return jsonify({"status": status, "predictions": results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)