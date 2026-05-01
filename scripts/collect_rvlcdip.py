"""
Télécharge un sous-ensemble de RVL-CDIP (documents scannés variés)
et l'enregistre dans data/raw/OTHER_DOC/ pour servir de classe négative.

RVL-CDIP contient 16 classes : invoice, form, email, handwritten, advertisement,
scientific_report, scientific_publication, specification, file_folder,
news_article, budget, invoice, presentation, questionnaire, resume, memo.

Usage :
  python scripts/collect_rvlcdip.py           # 300 images par défaut
  python scripts/collect_rvlcdip.py --limit 252
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "data" / "raw" / "OTHER_DOC"

# Classes RVL-CDIP à utiliser (on exclut rien, on veut de la variété)
LABEL_NAMES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "news_article", "budget", "invoice", "presentation",
    "questionnaire", "resume", "memo",
]


def collect(limit: int) -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(TARGET_DIR.glob("*.jpg")))
    if existing >= limit:
        log.info("Déjà %d images dans OTHER_DOC, rien à faire.", existing)
        return

    from datasets import load_dataset

    needed = limit - existing
    per_label = max(1, needed // len(LABEL_NAMES))
    log.info("Chargement RVL-CDIP (streaming)... %d images cibles.", needed)

    ds = load_dataset(
        "aharley/rvl_cdip",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    counts: dict[int, int] = {}
    saved = 0

    for sample in ds:
        if saved >= needed:
            break

        label = sample["label"]
        if counts.get(label, 0) >= per_label:
            continue

        img = sample["image"]
        dest = TARGET_DIR / f"rvlcdip_{existing + saved:04d}.jpg"
        try:
            img.convert("RGB").save(dest, "JPEG", quality=90)
            counts[label] = counts.get(label, 0) + 1
            saved += 1
            if saved % 50 == 0:
                log.info("%d / %d images sauvegardées...", saved, needed)
        except Exception as exc:
            log.debug("Erreur sauvegarde : %s", exc)

    log.info("Terminé : %d images ajoutées dans %s", saved, TARGET_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300, help="Nombre d'images cibles")
    args = parser.parse_args()
    collect(args.limit)
