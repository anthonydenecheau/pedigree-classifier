import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision
import json, os

# Optimisation VRAM 8Go
mixed_precision.set_global_policy('mixed_float16')

IMG_SIZE = (224, 224)
BATCH_SIZE = 16 

def train():
    with open('models/classes.json', 'r') as f:
        classes = json.load(f)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        'data/raw', validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        'data/raw', validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    ).prefetch(tf.data.AUTOTUNE)

    base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False)
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.RandomRotation(0.05),
        layers.Rescaling(1./127.5, offset=-1),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(len(classes), activation='softmax', dtype='float32')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    cb = [callbacks.EarlyStopping(patience=3, restore_best_weights=True),
          callbacks.ModelCheckpoint('models/best_pedigree_model.h5', save_best_only=True)]

    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=cb)
    print("✅ Entraînement terminé.")

if __name__ == "__main__": train()