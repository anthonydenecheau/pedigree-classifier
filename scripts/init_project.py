import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from countries import load_classes

ROOT = Path(__file__).resolve().parent.parent


def create_structure():
    base_dirs = ["data/raw", "data/processed", "data/inbox", "models", "scripts"]
    for d in base_dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    classes = load_classes()
    for class_name in classes:
        (ROOT / "data/raw" / class_name).mkdir(parents=True, exist_ok=True)

    print(f"Structure prête : {len(classes)} classes.")


if __name__ == "__main__":
    create_structure()