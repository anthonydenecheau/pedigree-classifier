VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
CONFIG=models/classes.json
MODEL=models/best_pedigree_model.h5

.PHONY: all init collect preprocess train evaluate serve clean check-data

all: init install collect preprocess train evaluate

init-venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: init-venv
	$(PIP) install tensorflow flask numpy pillow opencv-python playwright requests beautifulsoup4 seaborn matplotlib scikit-learn
	$(VENV)/bin/playwright install chromium

init:
	$(PYTHON) scripts/init_project.py

collect:
	$(PYTHON) scripts/collect_data.py

# Vérifie le nombre d'images par dossier avant de lancer l'entraînement
check-data:
	@echo "📊 État du dataset :"
	@ls -R data/raw | grep ":" | sed 's/://' | while read dir; do \
		count=$$(ls "$$dir" | wc -l); \
		echo "$$dir : $$count images"; \
	done

preprocess:
	$(PYTHON) scripts/preprocess.py

train:
	@if [ ! -f $(CONFIG) ]; then make init; fi
	$(PYTHON) scripts/train.py

# Nouvelle étape pour diagnostiquer les erreurs (Confusion)
evaluate:
	@if [ ! -f $(MODEL) ]; then echo "❌ Modèle introuvable. Lancez 'make train' d'abord."; exit 1; fi
	$(PYTHON) scripts/evaluate.py

serve:
	$(PYTHON) scripts/api.py

clean:
	rm -rf data/processed/*
	find . -type d -name "__pycache__" -exec rm -rf {} +