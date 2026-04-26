"""
Collecte de DOCUMENTS pedigree (certificats officiels scannés ou PDF).

Le modèle classifie des images de certificats d'enregistrement, pas des photos
de chiens. Les requêtes ciblent donc des scans de documents officiels.

Sources :
  1. Sites officiels des registres (PDF téléchargeables directement)
  2. Recherche Bing/DDG filtrée sur documents : "certificat" "scan" "pedigree"
  3. Flickr CC (scans de documents partagés par des éleveurs)

Usage :
  python scripts/collect_data.py              # collecte normale
  python scripts/collect_data.py --reset      # supprime data/raw/ et recommence
  python scripts/collect_data.py --limit 200  # nombre d'images cible par classe
"""

import argparse
import io
import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote_plus

import requests
from PIL import Image
from playwright.sync_api import sync_playwright

try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,de;q=0.7",
}

MIN_SIZE_BYTES = 20_000
TIMEOUT = 15
MAX_RETRIES = 3
SCROLL_COUNT = 5
WORKERS_DOWNLOAD = 8

# ---------------------------------------------------------------------------
# Requêtes ciblées sur les DOCUMENTS pedigree (pas les photos de chiens)
# ---------------------------------------------------------------------------

SEARCH_QUERIES: dict[str, list[str]] = {
    "LOF": [
        '"LOF" "certificat de naissance" scan pedigree',
        '"société centrale canine" "pedigree" certificat filetype:jpg',
        'site:chiens-de-france.com "pedigree" certificat document',
        '"LOF" "livre des origines français" document scan',
    ],
    "AKC": [
        '"AKC" "Certified Pedigree" document scan certificate',
        '"American Kennel Club" "registration certificate" scan filetype:jpg',
        '"AKC" pedigree certificate form official',
    ],
    "UKC": [
        '"United Kennel Club" "registration certificate" scan',
        '"UKC" pedigree certificate document official',
    ],
    "KC": [
        '"Kennel Club" "registration certificate" scan document UK',
        'site:thekennelclub.org.uk "pedigree" certificate official',
        '"KC" pedigree registration paper scan filetype:jpg',
    ],
    "KMSH": [
        '"KMSH" "pedigree" certificat document scan belgique',
        '"Koninklijke Maatschappij Sint-Hubertus" pedigree stamboom',
        '"LRSH" certificat naissance chien belgique scan',
    ],
    "VDH": [
        '"VDH" "Ahnentafel" Hund Dokument scan',
        '"Verband für das Deutsche Hundewesen" Ahnentafel offiziell',
        '"VDH" Rassehund Abstammungsnachweis filetype:pdf',
    ],
    "ENCI": [
        '"ENCI" "pedigree" certificato documento scan',
        '"Ente Nazionale Cinofilia Italiana" pedigree ufficiale',
        '"ENCI" attestato iscrizione LOI filetype:pdf',
    ],
    "RSCE": [
        '"RSCE" "pedigree" certificado documento scan',
        '"Real Sociedad Canina de España" pedigree oficial',
        '"RSCE" libro de origenes español pedigree',
    ],
    "SCS": [
        '"SKG" "SCG" Ahnentafel Hund Dokument scan',
        '"Schweizerische Kynologische Gesellschaft" pedigree',
        '"SCS" abstammungsurkunde hund filetype:pdf',
    ],
}

# URLs directes de pages officialles contenant des PDF ou exemples de documents
REGISTRY_DOCUMENT_URLS: dict[str, list[str]] = {
    "LOF":  ["https://www.centrale-canine.fr/le-lof/le-pedigree"],
    "AKC":  ["https://www.akc.org/register/information/pedigree/"],
    "KC":   ["https://www.thekennelclub.org.uk/registration/"],
    "VDH":  ["https://www.vdh.de/service/ahnentafel/"],
    "ENCI": ["https://www.enci.it/servizi/pedigree/"],
    "RSCE": ["https://www.rsce.es/servicios/pedigree/"],
    "KMSH": ["https://www.kmsh.be/fr/services/pedigree"],
    "SCS":  ["https://www.skg.ch/fr/services/pedigree/"],
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_valid_image(data: bytes) -> bool:
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        return img.format in ("JPEG", "PNG", "WEBP")
    except Exception:
        return False


def _is_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF"


def _pdf_to_images(data: bytes) -> list[Image.Image]:
    """Convertit chaque page d'un PDF en image PIL."""
    if not PDF_SUPPORT:
        log.warning("pdf2image non installé — PDFs ignorés. Installez : pip install pdf2image")
        return []
    try:
        return convert_from_bytes(data, dpi=200, fmt="jpeg", thread_count=2)
    except Exception as exc:
        log.debug("Erreur conversion PDF : %s", exc)
        return []

# ---------------------------------------------------------------------------
# Téléchargement avec retry
# ---------------------------------------------------------------------------

def _fetch_url(url: str) -> bytes | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                log.debug("Échec définitif %s : %s", url, exc)
            else:
                time.sleep(attempt * 1.5)
    return None

# ---------------------------------------------------------------------------
# Source 1 — Registres officiels (Playwright, cherche PDF + images de docs)
# ---------------------------------------------------------------------------

def _scrape_registry(class_name: str) -> list[str]:
    base_urls = REGISTRY_DOCUMENT_URLS.get(class_name, [])
    urls: list[str] = []
    if not base_urls:
        return urls
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            for base_url in base_urls:
                try:
                    page = browser.new_page(extra_http_headers=HEADERS)
                    page.goto(base_url, timeout=30_000)
                    time.sleep(2)
                    # Liens PDF
                    pdf_links = page.locator("a[href$='.pdf']").evaluate_all(
                        "links => links.map(a => a.href).filter(Boolean)"
                    )
                    urls.extend(pdf_links)
                    # Images de documents (exclure logos/icônes petits)
                    img_urls = page.locator("img").evaluate_all(
                        """imgs => imgs
                            .filter(i => i.naturalWidth > 300 && i.naturalHeight > 300)
                            .map(i => i.src)
                            .filter(s => s.startsWith('http'))"""
                    )
                    urls.extend(img_urls)
                    page.close()
                except Exception as exc:
                    log.debug("Registry %s : %s", base_url, exc)
            browser.close()
    except Exception as exc:
        log.warning("Playwright registry error %s : %s", class_name, exc)
    return _dedup(urls)

# ---------------------------------------------------------------------------
# Source 2 — Bing Images (requêtes document-focused)
# ---------------------------------------------------------------------------

def _scrape_bing(query: str) -> list[str]:
    urls: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(extra_http_headers=HEADERS)
            page.goto(
                f"https://www.bing.com/images/search?q={quote_plus(query)}&form=HDRSC2",
                timeout=30_000,
            )
            for _ in range(SCROLL_COUNT):
                page.mouse.wheel(0, 3000)
                time.sleep(1.5)
            for selector in ("img.mimg", "img[class*='mimg']", "a.iusc img", "div.imgpt img"):
                found = page.locator(selector).evaluate_all(
                    "imgs => imgs.map(i => i.src || i.dataset.src || '').filter(s => s.startsWith('http'))"
                )
                urls.extend(found)
            browser.close()
    except Exception as exc:
        log.warning("Bing error '%s' : %s", query, exc)
    return _dedup(urls)

# ---------------------------------------------------------------------------
# Source 3 — DuckDuckGo Images
# ---------------------------------------------------------------------------

def _scrape_duckduckgo(query: str) -> list[str]:
    urls: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(extra_http_headers=HEADERS)
            page.goto(
                f"https://duckduckgo.com/?q={quote_plus(query)}&iax=images&ia=images",
                timeout=30_000,
            )
            time.sleep(2)
            for _ in range(SCROLL_COUNT):
                page.mouse.wheel(0, 3000)
                time.sleep(1.2)
            found = page.locator("img.tile--img__img").evaluate_all(
                "imgs => imgs.map(i => i.src || i.dataset.src || '').filter(s => s.startsWith('http'))"
            )
            urls.extend(found)
            browser.close()
    except Exception as exc:
        log.warning("DuckDuckGo error '%s' : %s", query, exc)
    return _dedup(urls)

# ---------------------------------------------------------------------------
# Source 4 — Flickr CC (scans de documents partagés par des éleveurs)
# ---------------------------------------------------------------------------

FLICKR_QUERIES: dict[str, str] = {
    "LOF":  "LOF pedigree certificat document scan",
    "AKC":  "AKC pedigree certificate document scan",
    "UKC":  "UKC registration certificate document",
    "KC":   "Kennel Club pedigree certificate document scan",
    "KMSH": "pedigree stamboom belgique document",
    "VDH":  "VDH Ahnentafel Dokument scan",
    "ENCI": "ENCI pedigree documento certificato",
    "RSCE": "RSCE pedigree certificado documento",
    "SCS":  "SKG pedigree Ahnentafel Dokument",
}

def _scrape_flickr(class_name: str) -> list[str]:
    query = FLICKR_QUERIES.get(class_name, f"pedigree certificate document {class_name}")
    urls: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(extra_http_headers=HEADERS)
            flickr_url = (
                f"https://www.flickr.com/search/?text={quote_plus(query)}"
                "&license=1%2C2%2C3%2C4%2C5%2C6&media=photos"
            )
            page.goto(flickr_url, timeout=30_000)
            for _ in range(SCROLL_COUNT):
                page.mouse.wheel(0, 3000)
                time.sleep(1.2)
            found = page.locator("img.photo-list-photo-view").evaluate_all(
                "imgs => imgs.map(i => (i.src || i.dataset.src || '')).filter(s => s.startsWith('http'))"
            )
            urls.extend(found)
            browser.close()
    except Exception as exc:
        log.warning("Flickr error %s : %s", class_name, exc)
    return _dedup(urls)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for u in urls:
        if u and u.startswith("http") and u not in seen:
            seen.add(u)
            result.append(u)
    return result

# ---------------------------------------------------------------------------
# Sauvegarde (gère images et PDFs)
# ---------------------------------------------------------------------------

def _save_image(img: Image.Image, dest: str) -> bool:
    try:
        img.convert("RGB").save(dest, "JPEG", quality=92)
        return True
    except Exception as exc:
        log.debug("Erreur sauvegarde %s : %s", dest, exc)
        return False

# ---------------------------------------------------------------------------
# Orchestrateur principal
# ---------------------------------------------------------------------------

def collect_urls(class_name: str) -> list[str]:
    region = class_name.split("_", 1)[1] if "_" in class_name else class_name
    queries = SEARCH_QUERIES.get(region, [f'pedigree certificate document {region}'])
    all_urls: list[str] = []

    log.info("[%s] Scraping registres officiels...", class_name)
    all_urls.extend(_scrape_registry(class_name))

    log.info("[%s] Scraping Flickr CC (documents)...", class_name)
    all_urls.extend(_scrape_flickr(class_name))

    for q in queries:
        log.info("[%s] Bing : '%s'", class_name, q)
        all_urls.extend(_scrape_bing(q))
        log.info("[%s] DuckDuckGo : '%s'", class_name, q)
        all_urls.extend(_scrape_duckduckgo(q))

    deduped = _dedup(all_urls)
    log.info("[%s] %d URLs uniques.", class_name, len(deduped))
    return deduped


def download_images(class_name: str, save_path: str, limit: int = 300) -> int:
    os.makedirs(save_path, exist_ok=True)
    existing = len([f for f in os.listdir(save_path) if f.lower().endswith(".jpg")])
    if existing >= limit:
        log.info("[%s] Déjà %d images, skip.", class_name, existing)
        return 0

    all_urls = collect_urls(class_name)
    if not all_urls:
        log.warning("[%s] Aucune URL trouvée.", class_name)
        return 0

    count = existing

    with ThreadPoolExecutor(max_workers=WORKERS_DOWNLOAD) as pool:
        futures = {pool.submit(_fetch_url, url): url for url in all_urls}
        for future in as_completed(futures):
            if count >= limit:
                for f in futures:
                    f.cancel()
                break
            data = future.result()
            if not data:
                continue

            if _is_pdf(data):
                # Convertir chaque page du PDF en image
                pages = _pdf_to_images(data)
                for page_img in pages:
                    if count >= limit:
                        break
                    dest = os.path.join(save_path, f"img_{count:04d}.jpg")
                    if _save_image(page_img, dest):
                        count += 1
            elif len(data) > MIN_SIZE_BYTES and _is_valid_image(data):
                img = Image.open(io.BytesIO(data))
                dest = os.path.join(save_path, f"img_{count:04d}.jpg")
                if _save_image(img, dest):
                    count += 1

    added = count - existing
    log.info("[%s] %d images collectées (total %d / %d).", class_name, added, count, limit)
    return added

# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def reset_class(class_name: str, raw_dir: str = "data/raw") -> None:
    """Supprime uniquement les images du dossier, conserve le répertoire."""
    path = Path(raw_dir) / class_name
    path.mkdir(parents=True, exist_ok=True)
    deleted = 0
    for f in path.iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            f.unlink()
            deleted += 1
    log.info("[%s] %d images supprimées (dossier conservé).", class_name, deleted)


def reset_all(classes: list[str], raw_dir: str = "data/raw") -> None:
    """Supprime les images de toutes les classes, conserve les répertoires."""
    for class_name in classes:
        reset_class(class_name, raw_dir)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collecte de documents pedigree.")
    parser.add_argument(
        "--reset", action="store_true",
        help="Supprime toutes les images collectées avant de recommencer.",
    )
    parser.add_argument(
        "--class", dest="only_class", default=None,
        help="Collecte uniquement pour une classe (ex: FRA_LOF).",
    )
    parser.add_argument(
        "--limit", type=int, default=300,
        help="Nombre d'images cible par classe (défaut: 300).",
    )
    args = parser.parse_args()

    if not PDF_SUPPORT:
        log.warning(
            "pdf2image non disponible — les PDFs seront ignorés. "
            "Installez : pip install pdf2image  (nécessite poppler-utils)"
        )

    from countries import load_classes
    CLASSES = load_classes()

    target_classes = [args.only_class] if args.only_class else CLASSES

    if args.reset:
        if args.only_class:
            reset_class(args.only_class)
        else:
            reset_all(target_classes)
        log.info("Reset terminé. Lancez 'make collect' pour collecter.")
    else:
        for cls in target_classes:
            path = os.path.join("data/raw", cls)
            download_images(cls, path, limit=args.limit)
