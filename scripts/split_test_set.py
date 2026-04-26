"""
À lancer UNE SEULE FOIS après le tri manuel des images.
Déplace 15% des images de data/raw/ vers data/test/ (stratifié par classe).
Une fois fait, data/test/ est figé et ne doit jamais être utilisé pendant le training.
"""
import shutil
import pathlib
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SRC = pathlib.Path("data/raw")
DST = pathlib.Path("data/test")
TEST_SIZE = 0.15
SEED = 42


def split():
    if DST.exists() and any(DST.rglob("*.jpg")):
        log.warning("data/test/ existe déjà et contient des images. Abandon pour éviter l'écrasement.")
        return

    for cls_dir in sorted(SRC.iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        if len(files) < 2:
            log.warning("%s : pas assez d'images pour un split (%d), ignoré.", cls_dir.name, len(files))
            continue

        _, test_files = train_test_split(files, test_size=TEST_SIZE, random_state=SEED)
        dest_dir = DST / cls_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for f in test_files:
            shutil.move(str(f), dest_dir / f.name)

        log.info("%-30s : %d images déplacées vers data/test/", cls_dir.name, len(test_files))

    log.info("Split terminé. data/test/ est maintenant figé.")


if __name__ == "__main__":
    split()
