"""
Utilitaire partagé pour lire countries.json et dériver la liste des classes.
Source unique de vérité — plus besoin de models/classes.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COUNTRIES_FILE = ROOT / "countries.json"


def load_classes() -> list[str]:
    with open(COUNTRIES_FILE) as f:
        countries = json.load(f)
    classes = []
    for country, books in countries.items():
        # accepte "FRA": "LOF"  ou  "USA": ["AKC", "UKC"]
        if isinstance(books, str):
            books = [books]
        for book in books:
            classes.append(f"{country}_{book}")
    return classes
