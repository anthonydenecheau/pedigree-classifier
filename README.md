# 🐕 Pedigree Origin Detector (TensorFlow)

Projet de classification d'images pour détecter l'origine des pedigrees canins (Europe + USA).

## 📋 Rôle des Scripts
1. **init_project.py** : Crée l'arborescence et le fichier de classes à partir de `countries.json`.
2. **collect_data.py** : Récupère des scans via Bing/Google en utilisant des "Dorks" spécifiques.
3. **train.py** : Entraîne le modèle avec Transfer Learning (MobileNetV2) et optimisation 8Go VRAM.
4. **evaluate.py** : Génère une matrice de confusion pour voir quels pays l'IA confond.
5. **api.py** : Serveur Flask renvoyant le Top 3 des pays avec score et seuil de confiance.

## 🛠 Configuration Minimale
- **GPU** : NVIDIA RTX (8 Go VRAM minimum pour l'entraînement).
- **RAM** : 16 Go.
- **Logiciel** : Python 3.10+, Pilotes CUDA installés.

## 🚀 Lancement Rapide
```bash
make init      # Initialiser les dossiers
make collect   # Télécharger les images (Prend du temps)
# --- TRI MANUEL : Supprimer les mauvaises images dans data/raw ---
make train     # Entraîner l'IA
make evaluate  # Vérifier la précision
make serve     # Lancer l'API sur le port 5000