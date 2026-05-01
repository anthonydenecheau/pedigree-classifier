import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe, pas besoin de display X11
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

SEED = 42
TEST_DIR = "data/test"
RAW_DIR = "data/raw"


def evaluate():
    if not os.path.exists("models/best_pedigree_model.keras"):
        raise FileNotFoundError("Modèle introuvable. Lancez 'make train' d'abord.")

    model = tf.keras.models.load_model("models/best_pedigree_model.keras")

    # Mapping des indices du modèle vers les noms de classes
    class_names_path = "models/class_names.json"
    if not os.path.exists(class_names_path):
        raise FileNotFoundError("models/class_names.json introuvable. Relancez 'make train'.")
    with open(class_names_path) as f:
        model_classes = json.load(f)  # ordre alphabétique de data/raw au moment du train

    # Utilise data/test/ (jamais vu pendant le training) si disponible,
    # sinon repli sur le val split de data/raw/ (data leakage potentiel).
    if os.path.isdir(TEST_DIR) and any(
        f for f in os.scandir(TEST_DIR) if f.is_dir()
    ):
        print(f"Evaluation sur le test set : {TEST_DIR}")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            TEST_DIR, image_size=(224, 224), batch_size=32,
        )
    else:
        print(f"[ATTENTION] data/test/ absent — évaluation sur le val split de {RAW_DIR} (data leakage).")
        print("Lancez 'make split' pour créer un test set propre.")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            RAW_DIR, validation_split=0.2, subset="validation", seed=SEED,
            image_size=(224, 224), batch_size=32,
        )

    test_classes = val_ds.class_names
    print(f"Classes dans le test set ({len(test_classes)}) : {test_classes}")
    print(f"Classes du modèle ({len(model_classes)}) : {model_classes}")

    # Convertit les indices dataset → noms de classes → indices modèle
    test_to_model_idx = {i: model_classes.index(name) for i, name in enumerate(test_classes)}

    y_true_names, y_pred_names = [], []
    for imgs, labels in val_ds:
        preds = model.predict(imgs, verbose=0)
        pred_indices = np.argmax(preds, axis=1)
        for label, pred_idx in zip(labels.numpy(), pred_indices):
            y_true_names.append(test_classes[label])
            y_pred_names.append(model_classes[pred_idx])

    classes = sorted(set(y_true_names) | set(y_pred_names))

    # Métriques texte
    print(classification_report(y_true_names, y_pred_names, labels=classes, target_names=classes, zero_division=0))

    # Matrice de confusion sauvegardée sur disque
    cm = confusion_matrix(y_true_names, y_pred_names, labels=classes)
    plt.figure(figsize=(max(8, len(classes)), max(6, len(classes))))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.title("Matrice de confusion")
    plt.ylabel("Réel")
    plt.xlabel("Prédit")
    plt.tight_layout()
    out = "models/confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Matrice sauvegardée → {out}")


if __name__ == "__main__":
    evaluate()