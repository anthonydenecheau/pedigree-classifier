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
collect_data.py   →  scraping multi-sources : registres officiels, Flickr CC, Bing, DuckDuckGo
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
train.py          →  models/best_pedigree_model.h5
                      Phase 1 : feature extraction  (base MobileNetV2 gelée, LR=1e-3)
                      Phase 2 : fine-tuning 30 couches (LR=1e-5)
     │
     ▼
evaluate.py       →  models/confusion_matrix.png  +  classification_report
     │
     ▼
api.py            →  POST /predict   (Gunicorn, port 5000)
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

## Prérequis

- Python 3.12+
- NVIDIA GPU — 8 Go VRAM minimum (entraînement)
- CUDA + cuDNN installés
- `poppler-utils` pour la conversion PDF :
  ```bash
  sudo apt install poppler-utils
  ```

## Installation

```bash
make reinstall   # supprime le venv existant et réinstalle
# ou
make install     # installation initiale
```

## Pipeline complet

```bash
make init          # crée l'arborescence data/raw/{classe}/

make collect       # collecte des documents (300 images/classe, ~long)
make collect-reset # vide data/raw/ sans relancer la collecte

make label         # interface de labellisation → http://localhost:5001
                   # déposer les documents dans data/inbox/ avant de lancer

make split         # crée data/test/ (une seule fois, avant le premier train)
make preprocess    # valide et redimensionne les images → data/processed/

make train         # entraînement (2 phases : feature extraction + fine-tuning)
make evaluate      # matrice de confusion + métriques → models/confusion_matrix.png

make serve         # API de prédiction → http://localhost:5000
make check-data    # affiche le nombre d'images par classe
```

## API

```bash
# Prédiction
curl -X POST http://localhost:5000/predict \
     -F "file=@mon_pedigree.jpg"

# Réponse
{
  "status": "success",
  "predictions": [
    {"country": "FRA_LOF", "score": 87.4},
    {"country": "BEL_KMSH", "score": 7.2},
    {"country": "CHE_SCS",  "score": 3.1}
  ]
}

# Health check
curl http://localhost:5000/health
```

Le seuil de confiance est configurable :
```bash
PREDICT_THRESHOLD=65 make serve
```

## Interface de labellisation

```bash
# 1. Déposer les documents à classer
cp mes_scans/*.pdf data/inbox/

# 2. Lancer l'interface
make label   # → http://localhost:5001
```

- Raccourcis clavier : `1`–`9` pour classer, `R` pour rejeter
- Chaque document est déplacé dès validation → impossible de le classer deux fois
- Tableau de suivi (pays / livre / % objectif) affiché quand l'inbox est vide

## Configuration

| Fichier | Rôle |
|---------|------|
| `countries.json` | Source unique des pays et registres |
| `requirements.txt` | Dépendances Python avec versions épinglées |
| `scripts/countries.py` | Dérive la liste des classes depuis `countries.json` |

## Structure des données

```
data/
├── inbox/          ← documents à labelliser (déposer ici)
├── raw/            ← dataset labellisé par classe
│   ├── FRA_LOF/
│   ├── USA_AKC/
│   └── ...
├── processed/      ← images normalisées (224×224 JPEG)
└── test/           ← test set figé (créé par make split, ne pas modifier)

models/
├── best_pedigree_model.h5   ← modèle entraîné (gitignored)
└── confusion_matrix.png     ← dernière évaluation
```
