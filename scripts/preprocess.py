"""
Preprocessing : valide et normalise les images de data/raw/ vers data/processed/.
- Supprime les fichiers corrompus ou non-image
- Convertit en JPEG RGB 224×224
- Conserve la structure de dossiers par classe
"""
import os
import logging
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SRC = Path("data/raw")
DST = Path("data/processed")
IMG_SIZE = (224, 224)


def preprocess_class(class_dir: Path) -> tuple[int, int]:
    dst_dir = DST / class_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)

    ok = skipped = 0
    for src_file in class_dir.iterdir():
        if not src_file.is_file():
            continue
        dst_file = dst_dir / (src_file.stem + ".jpg")
        if dst_file.exists():
            ok += 1
            continue
        try:
            with Image.open(src_file) as img:
                img.verify()
            with Image.open(src_file) as img:
                img = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
                img.save(dst_file, "JPEG", quality=90)
            ok += 1
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            log.warning("Image invalide ignorée : %s (%s)", src_file.name, exc)
            skipped += 1

    return ok, skipped


def preprocess():
    if not SRC.exists():
        raise FileNotFoundError(f"Dossier source introuvable : {SRC}")

    total_ok = total_skip = 0
    for class_dir in sorted(SRC.iterdir()):
        if not class_dir.is_dir():
            continue
        ok, skip = preprocess_class(class_dir)
        log.info("%-30s : %d ok, %d ignorées", class_dir.name, ok, skip)
        total_ok += ok
        total_skip += skip

    log.info("Preprocessing terminé : %d images ok, %d ignorées.", total_ok, total_skip)


if __name__ == "__main__":
    preprocess()
