# Prochaines étapes

## Priorité 1 — Données (bloquant)

Sans données suffisantes pour les autres registres, le modèle ne peut classifier que FRA_LOF.

- [x] Collecter FRA_LOF (252 documents) via `import.sh`
- [x] Collecter la classe négative OTHER_DOC via `make collect-negative` (252 images RVL-CDIP)
- [x] Créer le test set (`make split`) — `data/test/FRA_LOF/` + `data/test/OTHER_DOC/`
- [x] Vérifier l'équilibre du dataset (`make check-data`)
- [ ] Collecter 200-300 documents pour les autres registres (GBR_KC, DEU_VDH, USA_AKC, etc.)
- [ ] Trier manuellement si nécessaire (`make label`)

Objectif final : **200-300 docs/registre × 9 registres + OTHER_DOC**

---

## Priorité 2 — Entraînement & évaluation

- [x] Premier entraînement complet (`make train`) — modèle binaire FRA_LOF vs OTHER_DOC
- [x] Analyser la matrice de confusion (`make evaluate`) — 100% accuracy
- [ ] Relancer `make train` une fois les autres registres alimentés
- [ ] Identifier les classes confondues et enrichir le dataset en conséquence
- [ ] Ajuster `FINE_TUNE_LAYERS` et `EPOCHS_PHASE2` si la val_accuracy plafonne

---

## Priorité 3 — Tracking des expériences

- [x] Intégrer MLflow dans `train.py` (`make mlflow` → http://localhost:5001)
- [x] Logger les hyperparamètres, métriques et artefacts à chaque run
- [ ] Comparer les runs phase 1 vs phase 1+2 pour valider le gain du fine-tuning
- [ ] Comparer le run binaire (2 classes) vs le run multi-classes (9+ classes)

---

## Priorité 4 — Versioning & reproductibilité

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

---

## Priorité 5 — Tests & CI/CD

- [ ] Créer `tests/test_api.py` (health check, predict valide/invalide, extension rejetée)
- [ ] Créer `tests/test_preprocess.py` (image corrompue rejetée, dimensions correctes)
- [ ] Ajouter un `Jenkinsfile` :
  ```
  Install → Lint (flake8) → Test (pytest) → Train (main) → Evaluate (main)
  ```
- [ ] Archiver `models/confusion_matrix.png` comme artefact Jenkins

---

## Priorité 6 — Production (après validation du modèle multi-classes)

- [ ] Passer `make serve` sur Gunicorn avec `-w` adapté au nombre de CPU
- [ ] Ajouter un `Dockerfile` pour containeriser l'API
- [ ] Mettre en place un reverse proxy (nginx) devant Gunicorn
- [ ] Ajouter une authentification API (clé ou JWT)
- [ ] Surveiller les prédictions en production (logs → Grafana ou ELK)

---

## Suivi

| Étape | Statut |
|-------|--------|
| Import manuel des PDFs pedigree (`import.sh`) | ✅ |
| Classe négative OTHER_DOC (RVL-CDIP, `make collect-negative`) | ✅ |
| Interface de labellisation (`make label`) | ✅ |
| Preprocessing + validation images (`make preprocess`) | ✅ |
| Test set figé (`make split`) | ✅ |
| Entraînement 2 phases MobileNetV2 (`make train`) | ✅ |
| Évaluation sur test set propre (`make evaluate`) | ✅ |
| API Flask + Gunicorn (`make serve`) | ✅ |
| Interface web drag & drop | ✅ |
| Tracking MLflow (`make mlflow`) | ✅ |
| Données autres registres (GBR_KC, DEU_VDH…) | ⏳ |
| Entraînement multi-classes | ⏳ |
| DVC — versioning du dataset | ⏳ |
| Tests unitaires | ⏳ |
| CI/CD Jenkins | ⏳ |
| Docker | ⏳ |
