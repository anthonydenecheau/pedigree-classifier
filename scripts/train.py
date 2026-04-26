import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Optimisation VRAM 8Go
mixed_precision.set_global_policy('mixed_float16')

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


FINE_TUNE_LAYERS = 30   # nombre de couches MobileNetV2 dégelées en phase 2
EPOCHS_PHASE1    = 15
EPOCHS_PHASE2    = 10


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Retourne (model_complet, base_model) pour garder une référence directe au tronc."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, name="mobilenetv2_trunk"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.10)(x)
    x = layers.RandomZoom(0.10)(x)
    x = layers.RandomBrightness(0.10)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    # training=False en phase 1 : BatchNorm utilise les stats ImageNet (plus stable)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs), base_model


def _make_callbacks() -> list:
    return [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint("models/best_pedigree_model.h5", save_best_only=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    ]


def train():
    from countries import load_classes
    classes = load_classes()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/raw", validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/raw", validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    )

    # Poids de classe pour compenser le déséquilibre du dataset
    labels = np.concatenate([y.numpy() for _, y in train_ds])
    class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weight_dict = dict(enumerate(class_weights))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

    model, base_model = build_model(len(classes))

    # ------------------------------------------------------------------
    # Phase 1 — Feature extraction : tronc MobileNetV2 gelé, LR=1e-3
    # ------------------------------------------------------------------
    print(f"Phase 1 : feature extraction ({EPOCHS_PHASE1} epochs max, LR=1e-3)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_PHASE1, callbacks=_make_callbacks(),
        class_weight=class_weight_dict,
    )

    # ------------------------------------------------------------------
    # Phase 2 — Fine-tuning : dégel des FINE_TUNE_LAYERS dernières couches
    # Les couches inférieures (features bas niveau) restent gelées.
    # LR très bas pour ne pas détruire les poids pré-entraînés.
    # BatchNorm en mode inference (training=True en appel de couche serait
    # instable avec BATCH_SIZE=16 ; on garde les stats ImageNet).
    # ------------------------------------------------------------------
    print(f"Phase 2 : fine-tuning des {FINE_TUNE_LAYERS} dernières couches (LR=1e-5)...")
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_PHASE2, callbacks=_make_callbacks(),
        class_weight=class_weight_dict,
    )

    print("Entraînement terminé.")


if __name__ == "__main__":
    train()