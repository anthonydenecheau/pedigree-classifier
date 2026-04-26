Tu es un expert senior en machine learning, MLOps et Python.

Je travaille sur un projet de classification d’images (chiens / pedigree) avec un pipeline basé sur :
- collecte de données (web scraping)
- preprocessing d’images
- entraînement d’un modèle TensorFlow (CNN)
- évaluation (matrice de confusion)
- exposition via API Flask

Voici mon Makefile et l’organisation actuelle du projet :

```bash
make init      # Initialiser les dossiers
make collect   # Télécharger les images (Prend du temps)
# --- TRI MANUEL : Supprimer les mauvaises images dans data/raw ---
make train     # Entraîner l'IA
make evaluate  # Vérifier la précision
make serve     # Lancer l'API sur le port 5000
```

Objectifs :
1. Auditer l’architecture globale (code + pipeline)
2. Identifier les faiblesses (robustesse, performance, dette technique)
3. Améliorer mon script pour la construction du dataset dans collect_data.py
4. Proposer une version améliorée "production-ready"
5. Suggérer des bonnes pratiques MLOps (reproductibilité, tracking, dataset versioning)
6. Donner des exemples concrets de code ou d’organisation

Contraintes :
- Projet simple à maintenir (pas de sur-engineering)
- Compatible Linux (Ubuntu)
- Utilisable en local + CI/CD (Jenkins)
- Python 3.10+

Je veux une réponse structurée avec :
- audit
- quick wins
- refactoring proposé
- roadmap d’amélioration
