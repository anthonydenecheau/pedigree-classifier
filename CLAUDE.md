# CLAUDE.md — Documentation technique du projet pedigree-classifier

## Vue d'ensemble

Classificateur de certificats pedigree canins par registre d'origine.
Entrée : scan d'un pedigree (JPG/PNG/WEBP). Sortie : registre détecté (FRA_LOF, GBR_KC, …) + score de confiance.

**Stack :** Python 3.12 · TensorFlow 2.19 · MobileNetV2 · Flask/Gunicorn · GPU RTX 4060 (8 Go VRAM)

---

## Structure du projet

```
pedigree/
├── countries.json              ← source unique des classes (pays + registres)
├── Makefile                    ← commandes du pipeline
├── requirements.txt
├── GPU_REQUIREMENTS.md         ← doc setup CUDA / tensorflow[and-cuda]
├── data/
│   ├── raw/                    ← dataset labellisé, une sous-dossier par classe
│   │   ├── FRA_LOF/            ← 252 images (page 2 des PDFs convertie en JPEG)
│   │   ├── OTHER_DOC/          ← classe négative (images RVL-CDIP)
│   │   └── {PAYS_REGISTRE}/    ← autres registres (vides pour l'instant)
│   ├── processed/              ← images redimensionnées 224×224 JPEG
│   ├── test/                   ← test set figé (créé une seule fois par make split)
│   └── inbox/                  ← documents à labelliser manuellement
├── models/
│   ├── best_pedigree_model.keras  ← modèle entraîné (gitignored)
│   ├── class_names.json           ← ordre des classes tel que vu à l'entraînement
│   └── confusion_matrix.png       ← dernière évaluation
└── scripts/
    ├── train.py
    ├── evaluate.py
    ├── api.py
    ├── preprocess.py
    ├── collect_rvlcdip.py
    ├── init_project.py
    ├── split_test_set.py
    ├── label.py
    ├── countries.py
    └── templates/
        └── index.html
```

---

## Commandes Makefile

| Commande | Description |
|----------|-------------|
| `make install` | Crée le venv et installe les dépendances |
| `make reinstall` | Supprime le venv et réinstalle |
| `make init` | Crée l'arborescence `data/raw/{classe}/` |
| `make preprocess` | Valide et redimensionne les images → `data/processed/` |
| `make split` | Crée `data/test/` (une seule fois avant le premier train) |
| `make train` | Entraînement (2 phases MobileNetV2) |
| `make evaluate` | Matrice de confusion + classification_report |
| `make serve` | API Flask + UI web sur http://localhost:5000 |
| `make check-data` | Compte les images par classe |
| `make label` | Interface de labellisation → http://localhost:5001 |

---

## Dataset

### Classes actives

| Classe | Images | Source |
|--------|--------|--------|
| `FRA_LOF` | 252 | PDFs pedigree importés via `import.sh`, page 2 extraite en JPEG |
| `OTHER_DOC` | 252 | RVL-CDIP (documents bureau variés) via `collect_rvlcdip.py` |
| Autres registres | 0 | Prévus mais pas encore collectés |

### Classe négative OTHER_DOC

Sans classe négative, le modèle attribuerait toujours un registre pedigree à n'importe quel document.
`OTHER_DOC` contient des scans issus de **RVL-CDIP** : lettres, formulaires, factures, articles, mémos, etc.
Si la prédiction top-1 est `OTHER_DOC`, l'API retourne `status: uncertain` (pas de faux registre).

**Collecte :**
```bash
python scripts/collect_rvlcdip.py --limit 252
```

### Import des pedigrees FRA_LOF

Les PDFs sont livrés en ZIP depuis `/home/anthony/projects/DATASET_PEDIGREE/`.
Le script `import.sh` dans ce dossier :
1. Détecte les ZIPs non en cours de copie (`lsof`)
2. Dézippe, extrait la **page 2** du PDF en JPEG (dpi=200, via `pdf2image`)
3. Sauvegarde dans `data/raw/FRA_LOF/`
4. Supprime le ZIP si succès

```bash
cd /home/anthony/projects/DATASET_PEDIGREE && bash import.sh
```

---

## Modèle

### Architecture : MobileNetV2 + transfer learning

Le modèle est un **MobileNetV2** pré-entraîné ImageNet avec une tête de classification custom.

```
Input (224×224×3)
  → Data augmentation (flip, rotation ±10°, zoom ±10%, brightness ±10%)
  → Rescaling (pixels [0,255] → [-1, 1])   ← normalization INTERNE au modèle
  → MobileNetV2 (tronc convolutif)
  → GlobalAveragePooling2D
  → Dropout(0.3)
  → Dense(num_classes, softmax)
```

**Point critique :** La normalisation est incluse dans le graphe du modèle via `Rescaling(1/127.5, offset=-1)`.
Ne jamais normaliser les pixels manuellement avant l'inférence — cela produit des prédictions erronées.

### Entraînement en 2 phases

**Phase 1 — Feature extraction** (`EPOCHS_PHASE1=15`, LR=1e-3)
- Tronc MobileNetV2 entièrement gelé (`base_model.trainable = False`)
- Seule la tête de classification est entraînée
- `BatchNorm` en mode inference (stats ImageNet conservées)

**Phase 2 — Fine-tuning** (`EPOCHS_PHASE2=10`, LR=1e-5)
- Les 30 dernières couches de MobileNetV2 sont dégelées (`FINE_TUNE_LAYERS=30`)
- LR très faible pour ne pas écraser les poids pré-entraînés
- `BatchNorm` reste en mode inference (instable avec `BATCH_SIZE=16` sinon)

### Optimisations GPU

- `mixed_precision.set_global_policy('mixed_float16')` — réduit la VRAM (~50%)
- `BATCH_SIZE=16` — adapté à 8 Go VRAM
- La couche Dense finale reste en `float32` pour la stabilité numérique du softmax

### Callbacks

| Callback | Rôle |
|----------|------|
| `EarlyStopping(patience=5)` | Arrête si val_accuracy stagne |
| `ModelCheckpoint` | Sauvegarde le meilleur modèle en `.keras` |
| `ReduceLROnPlateau(factor=0.5, patience=3)` | Réduit le LR sur plateau |

### Équilibrage des classes

`compute_class_weight("balanced")` de scikit-learn calcule des poids inversement proportionnels à la fréquence de chaque classe. Passés à `model.fit(class_weight=...)`.

---

## Fichier critique : `models/class_names.json`

Sauvegardé automatiquement par `train.py` au début de l'entraînement.
Contient l'ordre alphabétique des classes tel que vu par `image_dataset_from_directory`.

```json
["FRA_LOF", "OTHER_DOC"]
```

**Pourquoi c'est indispensable :**
- `image_dataset_from_directory` assigne des indices (0, 1, 2…) dans l'ordre alphabétique des dossiers
- L'indice 0 en training ≠ forcément indice 0 en évaluation si les dossiers diffèrent
- `evaluate.py` et `api.py` utilisent ce fichier pour convertir indices → noms de classes sans ambiguïté

---

## API

### Lancement

```bash
make serve   # gunicorn -w 1 -b 0.0.0.0:5000 scripts.api:app
```

**Important :** 1 seul worker (`-w 1`), pas de `--preload`.
Raison : avec `--preload`, le modèle TF est chargé dans le processus maître avant le fork.
Les workers héritent d'un contexte CUDA invalide → `CUDA_ERROR_NOT_INITIALIZED`.
Avec `-w 1` et chargement lazy, le modèle est initialisé dans le worker après le fork.

### Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Interface web (drag & drop) |
| GET | `/health` | Statut + nombre de classes |
| POST | `/predict` | Prédiction sur une image |

### Réponse `/predict`

```json
{
  "status": "success",
  "predictions": [
    {"country": "FRA_LOF", "score": 87.4},
    {"country": "OTHER_DOC", "score": 7.2},
    ...
  ]
}
```

`status` vaut `"success"` si `score[0] >= THRESHOLD` (défaut 50%), sinon `"uncertain"`.
Configurable : `PREDICT_THRESHOLD=65 make serve`

### Chargement lazy du modèle

```python
_model = None
_classes = None

def _get_model():
    global _model, _classes
    if _model is None:
        _model = tf.keras.models.load_model("models/best_pedigree_model.keras")
        with open("models/class_names.json") as f:
            _classes = json.load(f)
    return _model, _classes
```

Le modèle n'est chargé qu'à la première requête, jamais dans le processus maître Gunicorn.

---

## Setup GPU

Voir `GPU_REQUIREMENTS.md` pour le détail complet.

**Une seule commande suffit :**
```bash
pip install "tensorflow[and-cuda]"
```

Cette commande installe toutes les dépendances CUDA (cublas, cudnn, cufft…) avec les bonnes versions.
Ne pas installer les paquets `nvidia-*` manuellement + `LD_LIBRARY_PATH` → provoque des conflits
("already been registered") et TensorFlow ne détecte pas le GPU.

**Vérification :**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# → [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Pièges connus

### Double normalisation (bug critique)

Le modèle contient une couche `Rescaling(1/127.5, offset=-1)` en interne.
Si l'API normalise aussi les pixels avant l'inférence, les valeurs d'entrée sont hors plage → prédictions fausses.

**Correct dans `api.py` :**
```python
img_array = np.expand_dims(np.array(img), axis=0).astype("float32")
# NE PAS faire : (np.array(img) / 127.5) - 1.0
```

### Format du modèle

Utiliser le format `.keras` (TF 2.19), pas `.h5`.
Le format `.h5` est incompatible avec `mixed_float16` → erreur `cannot pickle 'module' object`.

### Décalage d'indices entre train et évaluation

Si `data/raw/` et `data/test/` n'ont pas les mêmes dossiers, les indices assignés par
`image_dataset_from_directory` diffèrent. Toujours comparer via les noms de classes,
jamais via les indices bruts. `class_names.json` est la référence.
