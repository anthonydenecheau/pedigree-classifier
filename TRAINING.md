# Workflow d'entraînement — `make train`

Guide détaillé du pipeline d'entraînement avec TensorFlow/Keras.

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
        ▼ Phase 1 : feature extraction (15 epochs max, LR=1e-3)
        │   tronc gelé, seule la tête s'entraîne
        │
        ▼ Phase 2 : fine-tuning (10 epochs max, LR=1e-5)
        │   30 dernières couches dégelées
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
  - ex : `FRA_LOF=0`, `OTHER_DOC=1` (si ce sont les seuls dossiers)
- Redimensionne toutes les images en 224×224 via bilinear interpolation
- Retourne des batches `(images, labels)` où `images` ∈ `[0, 255]` float32

**`validation_split=0.2` avec `seed=42` :**
Avec le même `seed`, le split train/val est déterministe et reproductible.
Les 20% de validation sont prélevés de façon stratifiée par dossier.

**Accès aux noms de classes :**
```python
num_classes = len(train_ds.class_names)
# train_ds.class_names → ["FRA_LOF", "OTHER_DOC"]
```

**Sauvegarde immédiate dans `class_names.json` :**
```python
with open("models/class_names.json", "w") as f:
    json.dump(train_ds.class_names, f)
```
Ce fichier est la référence pour `evaluate.py` et `api.py` — il garantit que l'indice 0
correspond toujours à la même classe, même si le dataset évolue.

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
Passé à `model.fit(class_weight=class_weight_dict)`.

**Pourquoi recalculer depuis les batches ?**
`image_dataset_from_directory` ne donne pas directement accès aux labels bruts.
La boucle `[y.numpy() for _, y in train_ds]` extrait tous les labels en one pass.

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

# Tronc MobileNetV2 (training=False en phase 1 → BatchNorm en mode inference)
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

# Sortie float32 pour stabilité numérique (mixed_float16 en vigueur)
outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
```

### Data augmentation intégrée

Les couches `Random*` sont des couches Keras standard, incluses dans le graphe.
Elles s'activent automatiquement en `training=True` (pendant `fit`) et sont désactivées
en `training=False` (pendant `predict` et `evaluate`).

| Couche | Effet |
|--------|-------|
| `RandomFlip("horizontal")` | Miroir horizontal (50% proba) |
| `RandomRotation(0.10)` | Rotation aléatoire ±36° max |
| `RandomZoom(0.10)` | Zoom ±10% |
| `RandomBrightness(0.10)` | Luminosité ±10% |

Ces augmentations simulent la variabilité des scans (orientation légèrement de travers,
luminosité du scanner différente, etc.).

### Rescaling — normalisation interne

```python
layers.Rescaling(1.0 / 127.5, offset=-1)
```

Transforme les pixels entiers [0, 255] vers [-1, 1] (format attendu par MobileNetV2).
Formule : `output = input × (1/127.5) + (-1)`

**Cette couche fait partie du modèle sauvegardé.** À l'inférence, passer les pixels bruts
(valeurs entières [0-255]). Ne jamais normaliser en dehors du modèle.

### `training=False` sur le tronc en phase 1

```python
x = base_model(x, training=False)
```

Force les couches `BatchNormalization` de MobileNetV2 à utiliser leurs statistiques
figées (moyennes/variances calculées sur ImageNet) même pendant l'entraînement.
Avec `BATCH_SIZE=16` et `training=True`, les stats de batch seraient trop bruitées
et déstabiliseraient l'entraînement.

### `dtype="float32"` sur la Dense finale

Avec `mixed_precision.set_global_policy('mixed_float16')`, toutes les couches utilisent
float16 par défaut. Le softmax sur float16 peut provoquer des underflows/overflows.
Forcer `float32` sur la couche de sortie garantit la stabilité numérique.

---

## 4. Mixed precision — optimisation mémoire GPU

```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

**Principe :**
- Les calculs (multiplications matricielles) se font en **float16** → 2× moins de VRAM
- Les gradients et les poids maîtres restent en **float32** → précision numérique préservée
- Gain typique : réduit la VRAM de ~40-50%, permet d'augmenter le batch size

**Avec une RTX 4060 Laptop (8 Go VRAM) :**
Sans mixed precision, BATCH_SIZE=16 avec MobileNetV2 peut dépasser la VRAM disponible.
Avec mixed precision, c'est confortable.

---

## 5. Phase 1 — Feature extraction

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=_make_callbacks(),
    class_weight=class_weight_dict,
)
```

**Objectif :** Entraîner uniquement la tête de classification.
Le tronc MobileNetV2 est gelé → ses poids ne changent pas.

**`sparse_categorical_crossentropy` :**
Utilisé quand les labels sont des entiers (0, 1, 2…) et non des vecteurs one-hot.
`image_dataset_from_directory` retourne des labels entiers → ce loss est adapté.

**`Adam(1e-3)` :**
LR élevé acceptable en phase 1 car seule la petite tête s'entraîne,
le risque de déstabiliser les poids pré-entraînés est nul (tronc gelé).

**EarlyStopping(patience=5) :**
Arrête l'entraînement si `val_accuracy` ne s'améliore pas pendant 5 epochs consécutives.
`restore_best_weights=True` ramène le modèle à son meilleur état.

**ReduceLROnPlateau(factor=0.5, patience=3) :**
Divise le LR par 2 si `val_loss` ne diminue plus pendant 3 epochs.
Aide à sortir des plateaux sans avoir à choisir manuellement un schedule.

---

## 6. Phase 2 — Fine-tuning

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:   # FINE_TUNE_LAYERS = 30
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=_make_callbacks(),
    class_weight=class_weight_dict,
)
```

**Objectif :** Adapter les couches hautes de MobileNetV2 au domaine des pedigrees.
Les couches basses (détection de bords, textures basiques) restent gelées —
elles sont génériques et utiles pour tout type d'image.
Les 30 dernières couches (features plus abstraites) sont dégelées pour apprendre
les patterns spécifiques aux pedigrees.

**`Adam(1e-5)` :**
LR 100× plus faible qu'en phase 1. Indispensable : un LR trop élevé écraserait
les poids pré-entraînés finement calibrés sur ImageNet.

**Pourquoi `model.compile()` à nouveau ?**
Après avoir modifié `trainable` sur des couches, il faut recompiler pour que
TensorFlow recalcule le graphe de gradient. Sans recompilation, les paramètres
dégelés ne seraient pas mis à jour.

---

## 7. Callbacks

```python
def _make_callbacks():
    return [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint("models/best_pedigree_model.keras", save_best_only=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    ]
```

Les trois callbacks sont appliqués aux deux phases.

**`ModelCheckpoint(save_best_only=True)` :**
Sauvegarde le modèle uniquement quand `val_accuracy` s'améliore.
Format `.keras` (natif TF 2.x) — obligatoire avec `mixed_float16`.
Le format `.h5` est incompatible et lève une erreur `cannot pickle 'module' object`.

---

## 8. Prefetch — optimisation de la pipeline de données

```python
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
```

`prefetch` prépare le batch suivant pendant que le GPU traite le batch courant.
`AUTOTUNE` laisse TensorFlow déterminer automatiquement le nombre de batches à précharger.
Sans `prefetch`, le GPU attend le CPU entre chaque batch → sous-utilisation GPU.

---

## 9. Évaluation — `make evaluate`

```python
# Chargement du test set (jamais vu pendant l'entraînement)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test", image_size=(224, 224), batch_size=32
)
test_classes = val_ds.class_names   # ordre alphabétique de data/test/

# Mapping : indices test → noms → indices modèle
with open("models/class_names.json") as f:
    model_classes = json.load(f)

y_true_names, y_pred_names = [], []
for imgs, labels in val_ds:
    preds = model.predict(imgs, verbose=0)
    pred_indices = np.argmax(preds, axis=1)
    for label, pred_idx in zip(labels.numpy(), pred_indices):
        y_true_names.append(test_classes[label])        # entier → nom
        y_pred_names.append(model_classes[pred_idx])    # entier → nom

# Comparaison via noms de classes (robuste aux décalages d'indices)
print(classification_report(y_true_names, y_pred_names, ...))
```

**Pourquoi comparer via les noms plutôt que les indices ?**
`data/test/` peut ne pas avoir exactement les mêmes dossiers que `data/raw/` au moment
de l'entraînement. L'ordre alphabétique change → `FRA_LOF` peut être l'indice 0 dans
le test set mais l'indice 4 dans le modèle. Passer par les noms de classes évite
tout décalage silencieux.

---

## 10. Résultats — run FRA_LOF

Modèle entraîné sur FRA_LOF + OTHER_DOC (252 images chacune).

```
Phase 1 (feature extraction) :
  val_accuracy ≈ 95-100% après 3-5 epochs
  EarlyStopping déclenché rapidement (tâche binaire simple)

Phase 2 (fine-tuning) :
  val_accuracy maintenu, val_loss légèrement réduite
```

**Classification report :**
```
              precision    recall  f1-score   support
     FRA_LOF       1.00      1.00      1.00        50
   OTHER_DOC       1.00      1.00      1.00        50
    accuracy                           1.00       100
```

100% sur le test set avec seulement 252 images par classe.
MobileNetV2 pré-entraîné est très efficace en few-shot pour les tâches binaires visuellement distinctes.

---

## Prochaines étapes pour étendre à d'autres registres

1. Collecter des images pour les autres classes (`GBR_KC`, `DEU_VDH`, etc.)
2. Atteindre ~200-300 images par classe
3. Relancer `make preprocess && make split && make train`
4. `class_names.json` sera mis à jour automatiquement
5. `OTHER_DOC` reste présent pour la robustesse en production
