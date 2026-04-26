# Audit & Roadmap MLOps — Pedigree Origin Detector

---

## 1. AUDIT DE L'ARCHITECTURE GLOBALE

### Pipeline actuel

```
countries.json
     │
     ▼
init_project.py   →  data/raw/{classe}/   models/classes.json
     │
     ▼
collect_data.py   →  scraping multi-sources (Bing, DuckDuckGo, Flickr CC, registres)
     │
     ▼  [TRI MANUEL]
     │
     ▼
split_test_set.py →  data/test/{classe}/   (une seule fois, avant le premier train)
     │
     ▼
preprocess.py     →  data/processed/{classe}/
     │
     ▼
train.py          →  models/best_pedigree_model.h5
                      Phase 1 : feature extraction (LR=1e-3, base gelée)
                      Phase 2 : fine-tuning 30 couches (LR=1e-5)
     │
     ▼
evaluate.py       →  models/confusion_matrix.png  +  classification_report
                      (sur data/test/, jamais vu pendant le training)
     │
     ▼
api.py            →  POST /predict  (Gunicorn 2 workers, :5000)
```

### Points forts

| Aspect | Évaluation |
|--------|-----------|
| Séparation des responsabilités (1 script = 1 étape) | ✅ Bonne base |
| MobileNetV2 pour contrainte VRAM 8 Go | ✅ Choix adapté |
| FP16 mixed precision | ✅ Optimisation utile |
| Early stopping + ReduceLROnPlateau | ✅ En place |
| Makefile pour orchestrer le pipeline | ✅ Reproductible localement |

### Faiblesses structurelles

| Aspect | Sévérité |
|--------|----------|
| 15 images pour 9 classes (6 classes vides) | 🔴 Bloquant — données à acquérir |
| Pas de `requirements.txt` (versions flottantes) | ✅ Corrigé |
| `preprocess.py` référencé mais absent | ✅ Corrigé |
| Data leakage : evaluate.py réutilise le val split du training | ✅ Corrigé |
| Pas de métriques (F1, recall, precision) | ✅ Corrigé |
| `plt.show()` incompatible headless/CI | ✅ Corrigé |
| API sans validation, sans logging, sans auth | ✅ Corrigé |
| `collect_data.py` : `except` bare, mono-thread, sélecteur fragile | ✅ Corrigé |

---

## 2. QUICK WINS — Appliqués

### `collect_data.py`
- ✅ `User-Agent` navigateur réel → évite la détection bot
- ✅ Retry avec back-off exponentiel (3 tentatives / URL)
- ✅ `ThreadPoolExecutor(max_workers=10)` → téléchargements parallèles
- ✅ Validation PIL (format + intégrité) avant sauvegarde
- ✅ 4 sélecteurs CSS Bing en cascade → résistance aux changements DOM
- ✅ Requêtes définies pour les 9 classes
- ✅ Reprise idempotente (skip si quota déjà atteint)
- ✅ Logging structuré (`logging` au lieu de `print`)
- ✅ 4 sources : registres officiels, Flickr CC, Bing, DuckDuckGo
- ✅ Limite cible : 300 images/classe

### `train.py`
- ✅ `SEED = 42` sur TensorFlow + NumPy → reproductibilité
- ✅ Augmentation enrichie : flip horizontal, rotation 10°, zoom 10°, luminosité ±10%
- ✅ `compute_class_weight("balanced")` → compense le déséquilibre
- ✅ `ReduceLROnPlateau` → convergence plus fine
- ✅ Phase 1 feature extraction + Phase 2 fine-tuning 30 couches (LR=1e-5)
- ✅ `build_model` retourne `(model, base_model)` → référence directe, pas d'index fragile
- ✅ `_make_callbacks()` instancie de nouveaux callbacks par phase → pas de fuite d'état
- ✅ `EarlyStopping(patience=5)`

### `evaluate.py`
- ✅ `matplotlib.use("Agg")` → headless/Jenkins sans display X11
- ✅ `plt.savefig()` → matrice persistée sur disque
- ✅ `classification_report` → precision, recall, F1 par classe
- ✅ Charge `data/test/` en priorité, repli sur val split avec avertissement
- ✅ Guard si le modèle est absent

### `api.py`
- ✅ Validation MIME (jpeg/png/webp uniquement)
- ✅ Limite upload 10 Mo (`MAX_CONTENT_LENGTH`)
- ✅ `try/except` ciblé sur ouverture image et inférence
- ✅ Logging de chaque prédiction
- ✅ Endpoint `/health`
- ✅ Seuil configurable via `PREDICT_THRESHOLD` (variable d'environnement)

### Fichiers créés
- ✅ `scripts/preprocess.py` — valide + redimensionne raw → processed
- ✅ `scripts/split_test_set.py` — crée `data/test/` (une seule fois, avant le premier train)
- ✅ `requirements.txt` — 12 dépendances avec versions épinglées

---

## 3. REFACTORING — Implémenté

### 3.1 `collect_data.py` — documents pedigree, multi-sources ✅

**Problème corrigé** : les requêtes génériques remontaient des photos de chiens.
Les requêtes sont maintenant ciblées sur les **certificats et documents officiels**
(termes : "certificat scan", "Ahnentafel Dokument", "registration certificate", filetype:pdf…).

Support PDF : les PDFs des registres officiels sont téléchargés et convertis
page par page en JPEG via `pdf2image` (nécessite `poppler-utils`).

```
collect_urls(class_name)
  ├── _scrape_registry()    → PDF + images de docs sur sites officiels
  ├── _scrape_flickr()      → scans de certificats CC partagés par éleveurs
  ├── _scrape_bing()        → requêtes "certificat scan" / "filetype:pdf"
  └── _scrape_duckduckgo()  → pool complémentaire
          │
          ▼
      _dedup()              → déduplication cross-sources
          │
          ▼
  ThreadPoolExecutor(8)     → téléchargements parallèles
          │
          ▼
  _is_pdf() → _pdf_to_images()   → conversion pages PDF → JPEG (dpi=200)
  _is_valid_image()              → PIL verify() + whitelist JPEG/PNG/WEBP
          │
          ▼
  PIL convert("RGB") .save()     → normalisation format sortie
```

Arguments CLI :
```bash
make collect                          # collecte normale (300 images/classe)
make collect-reset                    # reset data/raw/ puis collecte
python scripts/collect_data.py --reset --class EUROPE_FRANCE_SCC --limit 100
```

Prérequis système pour le support PDF :
```bash
sudo apt install poppler-utils        # requis par pdf2image
```

> Alternative si le scraping reste insuffisant : SerpAPI (payant, stable, zéro sélecteur à maintenir).

### 3.2 `train.py` — deux phases ✅

```
Phase 1 — Feature extraction  (EPOCHS_PHASE1=15, LR=1e-3)
  base_model.trainable = False
  BatchNorm en mode inference → stats ImageNet stables
  Objectif : adapter la tête Dense au domaine pedigree

Phase 2 — Fine-tuning  (EPOCHS_PHASE2=10, LR=1e-5)
  base_model.layers[:-FINE_TUNE_LAYERS].trainable = False
  30 dernières couches MobileNetV2 adaptées au domaine
  LR très bas → préserve les poids pré-entraînés des couches basses
```

Points techniques :
- `build_model()` retourne `(model, base_model)` → pas d'index `model.layers[N]` fragile
- `_make_callbacks()` crée de nouvelles instances par phase → évite la fuite d'état EarlyStopping entre les deux phases
- `FINE_TUNE_LAYERS`, `EPOCHS_PHASE1`, `EPOCHS_PHASE2` constants nommées en haut de fichier

### 3.3 `evaluate.py` — test set propre ✅

Workflow :
```
make collect → [tri manuel] → make split  (une seule fois)
                                   │
                                   ▼
                             data/test/  ← figé, jamais touché par train.py
                                   │
                              make evaluate
```

`evaluate.py` charge `data/test/` si disponible, sinon repli sur le val split
avec avertissement explicite. Métriques produites :
- `classification_report` (precision / recall / F1 / support par classe)
- `models/confusion_matrix.png` (sauvegardé, compatible CI)

### 3.4 `api.py` — Gunicorn en production ✅

```bash
make serve
# → gunicorn -w 2 -b 0.0.0.0:5000 --preload scripts.api:app
```

- `--preload` : le modèle est chargé **une seule fois** dans le process parent,
  puis partagé en mémoire par les 2 workers (copy-on-write). Évite de charger
  2× un modèle de plusieurs Go.
- 2 workers : 2 requêtes simultanées sans contention GPU.
- `PREDICT_THRESHOLD` configurable via variable d'environnement :
  `PREDICT_THRESHOLD=65 make serve`

---

## 4. ROADMAP MLOps

### Niveau 1 — Fondations (semaine 1-2)

**Objectif** : pipeline reproductible et données suffisantes.

```
1. Acquérir les données (bloquant absolu)
   → 300-500 images/classe minimum
   → make collect  (Flickr CC + registres + Bing + DDG)

2. Versionner les données avec DVC
   dvc init
   dvc add data/raw
   git add data/raw.dvc .dvcignore
   dvc remote add -d local_store /mnt/dvc-store

3. Créer le test set (une fois, puis figé)
   make split

4. Seeds fixes partout (déjà fait : SEED=42)
```

### Niveau 2 — Tracking des expériences (semaine 2-3)

**Objectif** : comparer les runs, ne jamais perdre un bon modèle.

```bash
pip install mlflow
```

```python
# Dans train.py, wrapper les deux model.fit :
import mlflow, mlflow.keras

with mlflow.start_run():
    mlflow.log_params({
        "seed": SEED, "batch_size": BATCH_SIZE,
        "base_model": "MobileNetV2", "fine_tune_layers": FINE_TUNE_LAYERS,
        "epochs_phase1": EPOCHS_PHASE1, "epochs_phase2": EPOCHS_PHASE2,
    })
    history1 = model.fit(...)   # phase 1
    history2 = model.fit(...)   # phase 2
    mlflow.log_metric("val_accuracy", max(history2.history["val_accuracy"]))
    mlflow.keras.log_model(model, "model")
```

```bash
mlflow ui   # → http://localhost:5000
```

### Niveau 3 — CI/CD Jenkins (semaine 3-4)

**Objectif** : valider chaque changement de code automatiquement.

```groovy
// Jenkinsfile
pipeline {
    agent any
    stages {
        stage('Install') {
            steps { sh 'make install' }
        }
        stage('Lint') {
            steps { sh '.venv/bin/python -m flake8 scripts/' }
        }
        stage('Test') {
            steps { sh '.venv/bin/pytest tests/' }
        }
        stage('Train') {
            when { branch 'main' }
            steps { sh 'make train' }
        }
        stage('Evaluate') {
            when { branch 'main' }
            steps { sh 'make evaluate' }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'models/confusion_matrix.png'
        }
    }
}
```

### Niveau 4 — Tests unitaires (en parallèle du niveau 3)

**Objectif** : détecter les régressions avant qu'elles arrivent en prod.

```
tests/
├── test_preprocess.py    → image corrompue rejetée, dimensions correctes
├── test_api.py           → /health 200, /predict avec image valide/invalide
└── test_collect.py       → _is_valid_image() sur JPEG/PNG/GIF/corrompu
```

```python
# tests/test_api.py
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json["status"] == "ok"

def test_predict_no_file(client):
    resp = client.post("/predict")
    assert resp.status_code == 400

def test_predict_invalid_mime(client):
    data = {"file": (b"fake", "x.txt", "text/plain")}
    resp = client.post("/predict", data=data)
    assert resp.status_code == 415
```

### Niveau 5 — Containerisation (optionnel, déploiement)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models/ models/
COPY scripts/api.py scripts/
EXPOSE 5000
CMD ["gunicorn", "-w", "2", "--preload", "-b", "0.0.0.0:5000", "scripts.api:app"]
```

```bash
docker build -t pedigree-api .
docker run -p 5000:5000 -e PREDICT_THRESHOLD=60 pedigree-api
```

---

## Résumé de la roadmap

| Semaine | Priorité | Action | Statut |
|---------|----------|--------|--------|
| S1 | 🔴 Bloquant | Acquérir 300-500 images/classe | ⏳ En attente |
| S1 | 🔴 Bloquant | `dvc init` + versionner `data/raw` | ⏳ En attente |
| S1 | ✅ Fait | Créer le test set figé (`data/test/`) | ✅ |
| S2 | 🟠 Majeur | Intégrer MLflow dans `train.py` | ⏳ En attente |
| S2 | 🟠 Majeur | Écrire les tests unitaires (api + preprocess) | ⏳ En attente |
| S3 | 🟡 Moyen | Jenkinsfile (lint → test → train → evaluate) | ⏳ En attente |
| S4 | ✅ Fait | Gunicorn en production (`make serve`) | ✅ |

---

> **Prérequis absolu avant tout** : tant que chaque classe n'a pas au moins 100 images
> (idéalement 300-500), les métriques du modèle ne sont pas interprétables.
> L'architecture du code est désormais saine — c'est la donnée qui est le vrai levier.
