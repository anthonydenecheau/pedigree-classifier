# Prochaines étapes

## Priorité 1 — Données (bloquant)

Sans données suffisantes, le modèle ne peut pas être évalué sérieusement.

- [ ] Collecter 300 documents minimum par registre (`make collect`)
- [ ] Trier manuellement les documents non pertinents (`make label`)
- [ ] Créer le test set une fois le dataset stabilisé (`make split`)
- [ ] Vérifier l'équilibre du dataset (`make check-data`)

Objectif : **300 docs/livre × 9 registres = 2 700 documents** dans `data/raw/`.

---

## Priorité 2 — Entraînement & évaluation

À faire une fois les données suffisantes.

- [ ] Lancer un premier entraînement complet (`make train`)
- [ ] Analyser la matrice de confusion (`make evaluate`)
- [ ] Identifier les classes confondues et enrichir le dataset en conséquence
- [ ] Ajuster `FINE_TUNE_LAYERS` et `EPOCHS_PHASE2` si la val_accuracy plafonne

---

## Priorité 3 — Versioning & reproductibilité

- [ ] Initialiser DVC pour versionner `data/raw/`
  ```bash
  pip install dvc
  dvc init
  dvc add data/raw
  git add data/raw.dvc .dvcignore
  ```
- [ ] Définir un remote DVC (NAS local, S3, Google Drive…)
  ```bash
  dvc remote add -d myremote /mnt/dvc-store
  dvc push
  ```
- [ ] Supprimer `models/classes.json` du repo (remplacé par `countries.json`)

---

## Priorité 4 — Tracking des expériences

- [ ] Intégrer MLflow dans `train.py`
  ```bash
  pip install mlflow
  mlflow ui   # → http://localhost:5000
  ```
- [ ] Logger les hyperparamètres, métriques et artefacts à chaque run
- [ ] Comparer les runs phase 1 vs phase 1+2 pour valider le gain du fine-tuning

---

## Priorité 5 — Tests & CI/CD

- [ ] Créer `tests/test_api.py` (health check, predict valide/invalide, MIME rejeté)
- [ ] Créer `tests/test_preprocess.py` (image corrompue rejetée, dimensions correctes)
- [ ] Créer `tests/test_collect.py` (`_is_valid_image` sur JPEG/PNG/GIF/corrompu)
- [ ] Ajouter un `Jenkinsfile` :
  ```
  Install → Lint (flake8) → Test (pytest) → Train (main) → Evaluate (main)
  ```
- [ ] Archiver `models/confusion_matrix.png` comme artefact Jenkins

---

## Priorité 6 — Amélioration de la collecte

- [ ] Évaluer la qualité des documents collectés : sont-ils bien des certificats ?
- [ ] Ajouter des sources pour les registres les moins alimentés (VDH, ENCI, RSCE)
- [ ] Envisager SerpAPI si le scraping Bing/DDG reste insuffisant
- [ ] Tester le téléchargement PDF sur les sites officiels (AKC, KC, SCC)

---

## Priorité 7 — Production (après validation du modèle)

- [ ] Passer `make serve` sur Gunicorn avec `--workers` adapté au nombre de CPU
- [ ] Ajouter un `Dockerfile` pour containeriser l'API
- [ ] Mettre en place un reverse proxy (nginx) devant Gunicorn
- [ ] Ajouter une authentification API (clé ou JWT)
- [ ] Surveiller les prédictions en production (logs → Grafana ou ELK)

---

## Suivi

| Étape | Statut |
|-------|--------|
| Pipeline de collecte multi-sources | ✅ |
| Interface de labellisation | ✅ |
| Preprocessing + validation images | ✅ |
| Entraînement 2 phases (MobileNetV2) | ✅ |
| Évaluation sur test set propre | ✅ |
| API Flask + Gunicorn | ✅ |
| Versioning `countries.json` comme source unique | ✅ |
| Collecte des données (300/livre) | ⏳ |
| Premier entraînement complet | ⏳ |
| DVC | ⏳ |
| MLflow | ⏳ |
| Tests unitaires | ⏳ |
| CI/CD Jenkins | ⏳ |
| Docker | ⏳ |
