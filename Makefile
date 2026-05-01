VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
CONFIG=countries.json
MODEL=models/best_pedigree_model.keras

.PHONY: all init collect-negative preprocess split train evaluate serve mlflow label clean clean-venv reinstall check-data

all: init install collect-negative preprocess train evaluate

init-venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: init-venv
	$(PIP) install -r requirements.txt
	$(VENV)/bin/playwright install chromium
	@which pdftoppm > /dev/null 2>&1 || (echo "⚠️  poppler-utils manquant (PDF). Installez : sudo apt install poppler-utils" && true)

init:
	$(PYTHON) scripts/init_project.py

# Collecte la classe négative OTHER_DOC depuis RVL-CDIP (HuggingFace streaming)
collect-negative:
	$(PYTHON) scripts/collect_rvlcdip.py --limit 300

# Vérifie le nombre d'images par dossier avant de lancer l'entraînement
check-data:
	@echo "📊 État du dataset :"
	@ls -R data/raw | grep ":" | sed 's/://' | while read dir; do \
		count=$$(ls "$$dir" | wc -l); \
		echo "$$dir : $$count images"; \
	done

preprocess:
	$(PYTHON) scripts/preprocess.py

# A lancer UNE SEULE FOIS apres le tri manuel, avant le premier training
split:
	@if [ -d data/test ] && [ "$$(find data/test -name '*.jpg' | wc -l)" -gt 0 ]; then \
		echo "data/test/ existe deja. Supprimez-le manuellement pour relancer."; exit 1; \
	fi
	$(PYTHON) scripts/split_test_set.py

train:
	@if [ ! -f $(CONFIG) ]; then make init; fi
	$(PYTHON) scripts/train.py

# Nouvelle étape pour diagnostiquer les erreurs (Confusion)
evaluate:
	@if [ ! -f $(MODEL) ]; then echo "❌ Modèle introuvable. Lancez 'make train' d'abord."; exit 1; fi
	$(PYTHON) scripts/evaluate.py

serve:
	$(VENV)/bin/gunicorn -w 1 -b 0.0.0.0:5000 scripts.api:app

mlflow:
	$(VENV)/bin/mlflow ui --port 5001

# Interface de labellisation manuelle (http://localhost:5001)
label:
	$(PYTHON) scripts/label.py

clean:
	rm -rf data/processed/*
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-venv:
	rm -rf $(VENV)
	@echo "Venv supprimé. Relancez : make install"

reinstall: clean-venv install