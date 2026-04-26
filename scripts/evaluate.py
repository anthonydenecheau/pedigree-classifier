import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json

def evaluate():
    model = tf.keras.models.load_model('models/best_pedigree_model.h5')
    with open('models/classes.json', 'r') as f:
        classes = json.load(f)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        'data/raw', validation_split=0.2, subset="validation", seed=123,
        image_size=(224, 224), batch_size=32
    )

    y_true, y_pred = [], []
    for imgs, labels in val_ds:
        preds = model.predict(imgs)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.show()

if __name__ == "__main__": evaluate()