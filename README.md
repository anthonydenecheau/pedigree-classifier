# pedigree-classifier

Classification de documents pedigree canins par registre d'origine (LOF, AKC, KC, VDH…)
via un CNN MobileNetV2 entraîné sur des scans de certificats officiels.

## Architecture

```
countries.json
     │
     ▼
init_project.py   →  data/raw/{classe}/   (arborescence par registre)
     │
     ▼
[Import manuel]   →  data/raw/{PAYS_REGISTRE}/   (PDFs → JPEG via import.sh)
collect_rvlcdip.py → data/raw/OTHER_DOC/         (classe négative, RVL-CDIP)
     │
     ▼  [Labellisation manuelle via make label]
     │
     ▼
split_test_set.py →  data/test/   (à lancer une seule fois, avant le premier entraînement)
     │
     ▼
preprocess.py     →  data/processed/   (validation + redimensionnement)
     │
     ▼
train.py          →  models/best_pedigree_model.keras
                      Phase 1 : feature extraction  (base MobileNetV2 gelée, LR=1e-3)
                      Phase 2 : fine-tuning 30 couches (LR=1e-5)
                      MLflow   : tracking automatique des métriques et artefacts
     │
     ▼
evaluate.py       →  models/confusion_matrix.png  +  classification_report
     │
     ▼
api.py            →  POST /predict   (Gunicorn 1 worker, port 5000)
                      GET  /          Interface web drag & drop
```

## Registres supportés

Définis dans `countries.json` — ajouter un pays = ajouter une ligne.

| Code ISO | Livre | Organisme |
|----------|-------|-----------|
| USA | AKC | American Kennel Club |
| USA | UKC | United Kennel Club |
| GBR | KC | The Kennel Club |
| FRA | LOF | Société Centrale Canine |
| DEU | VDH | Verband für das Deutsche Hundewesen |
| ITA | ENCI | Ente Nazionale Cinofilia Italiana |
| ESP | RSCE | Real Sociedad Canina de España |
| BEL | KMSH | Koninklijke Maatschappij Sint-Hubertus |
| CHE | SCS | Schweizerische Kynologische Gesellschaft |

## Classe négative — OTHER_DOC

Le modèle est entraîné en **classifieur multi-classes** : pour qu'il puisse rejeter
un document qui n'est pas un pedigree, il a besoin d'exemples de documents non-pedigree.

La classe `OTHER_DOC` remplit ce rôle. Elle est constituée d'images issues du dataset
public **RVL-CDIP** (Ryerson Vision Lab Complex Document Information Processing),
qui contient 400 000 scans de documents de bureau : factures, lettres, formulaires,
articles, budgets, mémos, etc.

```bash
python scripts/collect_rvlcdip.py --limit 252
```

Sans cette classe, le modèle prédirait toujours un registre pedigree même face à
un document quelconque — ce qui rendrait l'API inutilisable en production.

> Si le score le plus élevé appartient à `OTHER_DOC`, l'API retourne `status: uncertain`
> plutôt qu'un faux registre.

## Prérequis

- Python 3.12+
- NVIDIA GPU — 8 Go VRAM minimum (entraînement)
- `poppler-utils` pour la conversion PDF (utilisé par `import.sh`) :
  ```bash
  sudo apt install poppler-utils
  ```
- GPU : voir `GPU_REQUIREMENTS.md` — installer avec `pip install "tensorflow[and-cuda]"`

## Installation

```bash
make install     # crée le venv et installe toutes les dépendances
# ou
make reinstall   # supprime le venv existant et réinstalle
```

## Pipeline complet

```bash
make init              # crée l'arborescence data/raw/{classe}/

# Alimentation du dataset (deux sources)
# 1. Pedigrees : importer les PDFs via import.sh dans DATASET_PEDIGREE/
#    → extrait la page 2 en JPEG dans data/raw/{PAYS_REGISTRE}/
# 2. Classe négative OTHER_DOC :
make collect-negative  # télécharge ~300 images RVL-CDIP → data/raw/OTHER_DOC/

make label             # interface de labellisation → http://localhost:5001
                       # déposer les documents dans data/inbox/ avant de lancer

make split             # crée data/test/ (une seule fois, avant le premier train)
make preprocess        # valide et redimensionne les images → data/processed/

make train             # entraînement + tracking MLflow automatique
make evaluate          # matrice de confusion + métriques → models/confusion_matrix.png

make serve             # API de prédiction → http://localhost:5000 + interface UI
make mlflow            # tableau de bord des expériences → http://localhost:5001

make check-data        # affiche le nombre d'images par classe
```

## API

```bash
# Prédiction
curl -X POST http://localhost:5000/predict \
     -F "file=@mon_pedigree.jpg"

# Réponse — document reconnu
{
  "status": "success",
  "predictions": [
    {"country": "FRA_LOF", "score": 87.4},
    {"country": "OTHER_DOC", "score": 7.2},
    {"country": "GBR_KC",  "score": 3.1}
  ]
}

# Réponse — confiance insuffisante
{
  "status": "uncertain",
  "predictions": [...]
}

# Health check
curl http://localhost:5000/health
```

Le seuil de confiance est configurable :
```bash
PREDICT_THRESHOLD=65 make serve
```

## Tracking des expériences — MLflow

Chaque `make train` crée automatiquement un run MLflow dans l'expérience `pedigree-classifier`.

```bash
make mlflow   # → http://localhost:5001
```

Données trackées par run :
- **Paramètres** : seed, batch_size, base_model, fine_tune_layers, epochs, classes
- **Métriques** : val_accuracy et val_loss par phase + `final_val_accuracy`
- **Artefacts** : `class_names.json`, `confusion_matrix.png`, modèle Keras complet

## Interface de labellisation

```bash
cp mes_scans/*.pdf data/inbox/
make label   # → http://localhost:5001
```

- Raccourcis clavier : `1`–`9` pour classer, `R` pour rejeter
- Chaque document est déplacé dès validation → impossible de le classer deux fois

## Configuration

| Fichier | Rôle |
|---------|------|
| `countries.json` | Source unique des pays et registres |
| `requirements.txt` | Dépendances Python avec versions épinglées |
| `scripts/countries.py` | Dérive la liste des classes depuis `countries.json` |
| `scripts/collect_rvlcdip.py` | Collecte la classe négative OTHER_DOC depuis RVL-CDIP |
| `GPU_REQUIREMENTS.md` | Setup CUDA / tensorflow[and-cuda] |
| `/home/anthony/projects/DATASET_PEDIGREE/import.sh` | Import des PDFs pedigree → JPEG |

## Structure des données

```
data/
├── inbox/          ← documents à labelliser (déposer ici)
├── raw/            ← dataset labellisé par classe
│   ├── FRA_LOF/    ← 252 images (page 2 des PDFs convertie en JPEG)
│   ├── OTHER_DOC/  ← 252 images (RVL-CDIP)
│   └── {PAYS_REGISTRE}/   ← autres registres (vides, en attente de données)
├── processed/      ← images normalisées 224×224 JPEG
├── rejected/       ← images rejetées lors de la labellisation
└── test/           ← test set figé (créé par make split, ne pas modifier)
    ├── FRA_LOF/
    └── OTHER_DOC/

models/
├── best_pedigree_model.keras  ← modèle entraîné (gitignored)
├── class_names.json           ← ordre des classes (généré par train.py)
└── confusion_matrix.png       ← dernière évaluation

mlruns/                        ← base MLflow (gitignored)
```
