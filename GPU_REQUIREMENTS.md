# Configuration GPU pour TensorFlow

## Environnement

| Composant | Version |
|-----------|---------|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 Go VRAM) |
| Driver NVIDIA | 580.126.09 |
| CUDA (driver) | 13.0 |
| TensorFlow | 2.19.1 |
| Python | 3.12 |
| OS | Ubuntu / Linux x86_64 |

---

## Ce qui ne fonctionne PAS

### Installation manuelle des paquets nvidia-*

Installer séparément `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, etc. puis configurer
`LD_LIBRARY_PATH` manuellement dans le script `activate` du venv **ne suffit pas**.
TensorFlow 2.19 détecte les librairies mais échoue à enregistrer ses plugins CUDA
(erreur "already been registered") et ne voit pas le GPU.

---

## Ce qui fonctionne

### Une seule commande

```bash
pip install "tensorflow[and-cuda]"
```

`tensorflow[and-cuda]` installe automatiquement toutes les dépendances CUDA nécessaires
avec les bonnes versions et configure les chemins sans modifier `LD_LIBRARY_PATH` :

| Paquet installé | Version |
|-----------------|---------|
| nvidia-cublas-cu12 | 12.x |
| nvidia-cuda-cupti-cu12 | 12.x |
| nvidia-cuda-nvcc-cu12 | 12.x |
| nvidia-cuda-nvrtc-cu12 | 12.x |
| nvidia-cuda-runtime-cu12 | 12.x |
| nvidia-cudnn-cu12 | 9.x |
| nvidia-cufft-cu12 | 11.x |
| nvidia-cusolver-cu12 | 11.x |
| nvidia-cusparse-cu12 | 12.x |
| nvidia-nccl-cu12 | 2.x |
| nvidia-nvjitlink-cu12 | 12.x |

> Aucune modification de `LD_LIBRARY_PATH` ou du script `activate` n'est nécessaire.

---

## Vérification

```bash
source .venv/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# → [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Prérequis système

- Driver NVIDIA installé (`nvidia-smi` fonctionnel)
- **Pas besoin** d'installer le CUDA Toolkit système (cuda-toolkit, libcudnn-dev…)
  — tout est géré dans le venv par `tensorflow[and-cuda]`
