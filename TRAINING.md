# Workflow d'entraînement — `make train`

Guide détaillé du pipeline d'entraînement avec TensorFlow/Keras et MLflow.

---

## Vue d'ensemble du pipeline

```
data/raw/{classe}/          ← images brutes labellisées
        │
        ▼ image_dataset_from_directory (split 80/20)
train_ds / val_ds           ← tf.data.Dataset avec labels entiers
        │
        ▼ compute_class_weight
class_weight_dict           ← poids pour compenser le déséquilibre
        │
        ▼ build_model(num_classes)
MobileNetV2 + tête custom   ← graphe Keras
        │
        ▼ mlflow.start_run()  ←─────────────────────────────────┐
        │                                                        │
        ▼ Phase 1 : feature extraction (15 epochs max, LR=1e-3) │
        │   tronc gelé, seule la tête s'entraîne                │
        │   → log_metrics(phase1_*)                             │
        │                                                        │
        ▼ Phase 2 : fine-tuning (10 epochs max, LR=1e-5)        │
        │   30 dernières couches dégelées                        │
        │   → log_metrics(phase2_*, final_val_accuracy)         │
        │                                                        │
        ▼ log_artifacts(class_names.json, confusion_matrix.png) │
        ▼ log_model(model)  ────────────────────────────────────┘
        │
        ▼ ModelCheckpoint
models/best_pedigree_model.keras
models/class_names.json
```

---

## 1. Chargement du dataset avec `image_dataset_from_directory`

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/raw",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(224, 224),
    batch_size=16,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/raw",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(224, 224),
    batch_size=16,
)
```

**Ce que fait cette fonction :**
- Scanne `data/raw/` et détecte les sous-dossiers → une classe par dossier
- Assigne des entiers dans l'**ordre alphabétique** des noms de dossiers
  - ex : `FRA_LOF=0`, `OTHER_DOC=1`
- Redimensionne toutes les images en 224×224 via bilinear interpolation
- Retourne des batches `(images, labels)` où `images` ∈ `[0, 255]` float32

**`validation_split=0.2` avec `seed=42` :**
Split déterministe et reproductible. Les 20% de validation sont prélevés de façon
stratifiée par dossier — le même seed garantit les mêmes images en train/val à chaque run.

**Sauvegarde immédiate dans `class_names.json` :**
```python
with open("models/class_names.json", "w") as f:
    json.dump(train_ds.class_names, f)
```
Ce fichier est la référence pour `evaluate.py` et `api.py`.

---

## 2. Équilibrage des classes avec `compute_class_weight`

Quand les classes ont des effectifs inégaux, le modèle est biaisé vers la classe majoritaire.

```python
labels = np.concatenate([y.numpy() for _, y in train_ds])
unique_labels = np.unique(labels)
class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
class_weight_dict = dict(zip(unique_labels.tolist(), class_weights))
```

**Formule `"balanced"` :**
```
weight[i] = n_samples / (n_classes × count[i])
```

Si FRA_LOF a 200 images et OTHER_DOC en a 100 :
- `weight[FRA_LOF]` = 300 / (2 × 200) = 0.75
- `weight[OTHER_DOC]` = 300 / (2 × 100) = 1.50

Le modèle pénalise davantage les erreurs sur la classe minoritaire.

---

## 3. Construction du modèle — `build_model(num_classes)`

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    name="mobilenetv2_trunk"
)
base_model.trainable = False
```

`include_top=False` charge MobileNetV2 sans la tête de classification ImageNet (1000 classes).
On conserve uniquement le tronc convolutif → feature extractor.

### Graphe complet

```python
inputs = layers.Input(shape=(224, 224, 3))

# Data augmentation (actif seulement pendant l'entraînement)
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.10)(x)
x = layers.RandomZoom(0.10)(x)
x = layers.RandomBrightness(0.10)(x)

# Normalisation MobileNetV2 : [0,255] → [-1, 1]
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

# Tronc MobileNetV2 (training=False → BatchNorm en mode inference)
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

# float32 pour stabilité numérique (mixed_float16 en vigueur)
outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
```

### Data augmentation intégrée

Les couches `Random*` s'activent automatiquement en `training=True` (pendant `fit`)
et sont désactivées en `training=False` (pendant `predict` et `evaluate`).

| Couche | Effet |
|--------|-------|
| `RandomFlip("horizontal")` | Miroir horizontal (50% proba) |
| `RandomRotation(0.10)` | Rotation aléatoire ±36° max |
| `RandomZoom(0.10)` | Zoom ±10% |
| `RandomBrightness(0.10)` | Luminosité ±10% |

### Rescaling — normalisation interne

```python
layers.Rescaling(1.0 / 127.5, offset=-1)
```

Transforme les pixels entiers [0, 255] vers [-1, 1] (format attendu par MobileNetV2).
**Cette couche fait partie du modèle sauvegardé.** À l'inférence, passer les pixels bruts.

### `training=False` sur le tronc en phase 1

Force les `BatchNormalization` de MobileNetV2 à utiliser leurs statistiques figées (ImageNet).
Avec `BATCH_SIZE=16` et `training=True`, les stats de batch seraient trop bruitées.

### `dtype="float32"` sur la Dense finale

Avec `mixed_float16`, toutes les couches utilisent float16 par défaut.
Le softmax sur float16 peut provoquer des underflows. Forcer `float32` sur la sortie garantit la stabilité.

---

## 4. Mixed precision — optimisation mémoire GPU

```python
mixed_precision.set_global_policy('mixed_float16')
```

- Les calculs se font en **float16** → ~50% moins de VRAM
- Les gradients et poids maîtres restent en **float32** → précision préservée
- Sur RTX 4060 Laptop (8 Go VRAM) : permet `BATCH_SIZE=16` confortablement

---

## 5. Tracking MLflow

Chaque run est encapsulé dans un contexte `mlflow.start_run()`.

```python
mlflow.set_experiment("pedigree-classifier")
with mlflow.start_run():
    mlflow.log_params({...})   # avant l'entraînement

    history1 = model.fit(...)  # phase 1
    mlflow.log_metrics({
        "phase1_best_val_accuracy": max(history1.history["val_accuracy"]),
        "phase1_best_val_loss": min(history1.history["val_loss"]),
        "phase1_epochs_run": len(history1.history["val_accuracy"]),
    })

    history2 = model.fit(...)  # phase 2
    mlflow.log_metrics({
        "phase2_best_val_accuracy": ...,
        "final_val_accuracy": max(phase1_best, phase2_best),
    })

    mlflow.log_artifact("models/class_names.json")
    mlflow.tensorflow.log_model(model, artifact_path="model")
```

**`phase1_epochs_run` / `phase2_epochs_run` :**
`EarlyStopping` peut interrompre avant le nombre max d'epochs.
Logger le nombre réel permet de détecter les runs qui convergent vite vs ceux qui n'ont pas assez d'epochs.

**`final_val_accuracy` :**
Prend le meilleur score des deux phases. Permet de trier les runs dans l'UI MLflow
par une métrique unique indépendante du fait que la phase 2 améliore ou non la phase 1.

---

## 6. Phase 1 — Feature extraction

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=15, callbacks=_make_callbacks(),
    class_weight=class_weight_dict,
)
```

**`sparse_categorical_crossentropy` :**
Utilisé quand les labels sont des entiers (0, 1, 2…) et non des vecteurs one-hot.
`image_dataset_from_directory` retourne des labels entiers.

**`Adam(1e-3)` :**
LR élevé acceptable : seule la petite tête s'entraîne, le tronc est gelé.

---

## 7. Phase 2 — Fine-tuning

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history2 = model.fit(...)
```

**Pourquoi `model.compile()` à nouveau ?**
Après avoir modifié `trainable` sur des couches, il faut recompiler pour que
TensorFlow recalcule le graphe de gradient. Sans recompilation, les paramètres
dégelés ne seraient pas mis à jour.

**`Adam(1e-5)` :**
LR 100× plus faible qu'en phase 1. Indispensable pour ne pas écraser les poids
pré-entraînés finement calibrés sur ImageNet.

**Les 30 dernières couches seulement :**
Les couches basses de MobileNetV2 (détection de bords, textures basiques) restent gelées.
Elles sont génériques et utiles pour tout type d'image.
Les couches hautes (features plus abstraites) sont adaptées au domaine des pedigrees.

---

## 8. Callbacks

```python
def _make_callbacks():
    return [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint("models/best_pedigree_model.keras", save_best_only=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    ]
```

Les callbacks sont **réinstanciés** pour chaque phase via `_make_callbacks()`.
Si on réutilisait les mêmes instances, l'état interne de `EarlyStopping` (compteur de patience,
meilleure valeur connue) serait hérité de la phase 1 → risque d'arrêt immédiat en phase 2.

**`ModelCheckpoint(save_best_only=True)` :**
Format `.keras` obligatoire avec `mixed_float16`. Le format `.h5` est incompatible.

**`ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)` :**
Divise le LR par 2 si `val_loss` ne diminue plus pendant 3 epochs.
`min_lr=1e-6` empêche le LR de descendre en dessous d'un seuil inutilisable.

---

## 9. Prefetch — optimisation de la pipeline de données

```python
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
```

Prépare le batch suivant pendant que le GPU traite le batch courant.
Sans `prefetch`, le GPU attend le CPU entre chaque batch → sous-utilisation GPU.
`AUTOTUNE` laisse TensorFlow déterminer automatiquement le nombre de batches à précharger.

---

## 10. Évaluation — `make evaluate`

```python
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test", image_size=(224, 224), batch_size=32
)
test_classes = val_ds.class_names

with open("models/class_names.json") as f:
    model_classes = json.load(f)

y_true_names, y_pred_names = [], []
for imgs, labels in val_ds:
    preds = model.predict(imgs, verbose=0)
    pred_indices = np.argmax(preds, axis=1)
    for label, pred_idx in zip(labels.numpy(), pred_indices):
        y_true_names.append(test_classes[label])
        y_pred_names.append(model_classes[pred_idx])

print(classification_report(y_true_names, y_pred_names, ...))
```

**Pourquoi comparer via les noms plutôt que les indices ?**
`data/test/` peut ne pas avoir exactement les mêmes dossiers que `data/raw/` au moment
de l'entraînement. L'ordre alphabétique change → `FRA_LOF` peut être l'indice 0 dans
le test set mais l'indice 4 dans le modèle. Passer par les noms évite tout décalage silencieux.

---

## 11. Résultats — run FRA_LOF (mai 2026)

Modèle binaire FRA_LOF vs OTHER_DOC (252 images chacune).

| Métrique MLflow | Valeur |
|----------------|--------|
| `phase1_best_val_accuracy` | ~1.00 |
| `phase1_epochs_run` | 3–5 (EarlyStopping) |
| `phase2_best_val_accuracy` | ~1.00 |
| `final_val_accuracy` | 1.00 |

**Classification report sur `data/test/` :**
```
              precision    recall  f1-score   support
     FRA_LOF       1.00      1.00      1.00        ~50
   OTHER_DOC       1.00      1.00      1.00        ~50
    accuracy                           1.00       ~100
```

100% sur le test set avec seulement 252 images par classe.
MobileNetV2 pré-entraîné est très efficace en few-shot pour les tâches binaires
où les deux classes sont visuellement très distinctes.

---

## 12. Prochaines étapes pour étendre à d'autres registres

1. Collecter des images pour `GBR_KC`, `DEU_VDH`, `USA_AKC`, etc. (200–300 images/classe)
2. Relancer `make preprocess && make split && make train`
3. `class_names.json` sera mis à jour automatiquement
4. MLflow permettra de comparer le nouveau run multi-classes avec le run binaire
5. `OTHER_DOC` reste présent pour la robustesse en production
