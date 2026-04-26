import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe, pas besoin de display X11
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from countries import load_classes

SEED = 42
TEST_DIR = "data/test"
RAW_DIR = "data/raw"


def evaluate():
    if not os.path.exists("models/best_pedigree_model.h5"):
        raise FileNotFoundError("Modèle introuvable. Lancez 'make train' d'abord.")

    model = tf.keras.models.load_model("models/best_pedigree_model.h5")
    classes = load_classes()

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

    y_true, y_pred = [], []
    for imgs, labels in val_ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    # Métriques texte
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    # Matrice de confusion sauvegardée sur disque
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9))
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