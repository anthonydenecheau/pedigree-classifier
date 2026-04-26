"""
Interface de labellisation des documents pedigree.

Flux :
  data/inbox/   →  [interface web]  →  data/raw/<classe>/   (classifié)
                                    →  data/rejected/        (rejeté)

Déduplication : chaque fichier est déplacé dès validation — impossible de le
classer deux fois même après redémarrage du serveur.

Usage :
  python scripts/label.py
  → http://localhost:5001
"""

import io
import json
import logging
import os
import shutil
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template_string, request, send_file, url_for

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from countries import load_classes as _load_classes

ROOT     = Path(__file__).resolve().parent.parent
INBOX    = ROOT / "data/inbox"
RAW      = ROOT / "data/raw"
REJECTED = ROOT / "data/rejected"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}
MIN_DOCS_TARGET = 300   # seuil minimum par livre pour lancer l'entraînement

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_classes() -> list[str]:
    return _load_classes()


def inbox_files() -> list[Path]:
    if not INBOX.exists():
        return []
    return sorted(
        f for f in INBOX.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    )


def stats() -> dict[str, int]:
    classes = load_classes()
    return {
        cls: len(list((RAW / cls).glob("*.*"))) if (RAW / cls).exists() else 0
        for cls in classes
    }


def stats_by_country() -> list[dict]:
    """Retourne les stats groupées par pays, triées par total décroissant.

    [
      { "country": "FRA", "total": 12, "books": [{"name": "LOF", "count": 12}] },
      ...
    ]
    """
    flat = stats()
    groups: dict[str, list[dict]] = {}
    for cls, count in flat.items():
        country, book = cls.split("_", 1)
        groups.setdefault(country, []).append({"name": book, "count": count})

    result = []
    for country, books in groups.items():
        result.append({
            "country": country,
            "total": sum(b["count"] for b in books),
            "books": books,
        })
    result.sort(key=lambda x: x["total"], reverse=True)
    return result


def next_index(save_dir: Path) -> int:
    existing = list(save_dir.glob("img_*.jpg")) + list(save_dir.glob("img_*.png"))
    return len(existing)


def move_to_class(src: Path, class_name: str) -> None:
    dest_dir = RAW / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    idx = next_index(dest_dir)
    suffix = ".pdf" if src.suffix.lower() == ".pdf" else ".jpg"
    dest = dest_dir / f"img_{idx:04d}{suffix}"
    shutil.move(str(src), dest)
    log.info("Classifié : %s → %s", src.name, dest)


def move_to_rejected(src: Path) -> None:
    REJECTED.mkdir(parents=True, exist_ok=True)
    dest = REJECTED / src.name
    # évite les collisions de noms dans rejected/
    if dest.exists():
        dest = REJECTED / f"{src.stem}_{os.urandom(4).hex()}{src.suffix}"
    shutil.move(str(src), dest)
    log.info("Rejeté : %s", src.name)


def pdf_first_page_jpeg(path: Path) -> bytes:
    """Renvoie la première page d'un PDF en JPEG bytes pour la prévisualisation."""
    if not PDF_SUPPORT:
        return b""
    pages = convert_from_path(str(path), dpi=150, first_page=1, last_page=1)
    buf = io.BytesIO()
    pages[0].save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Template HTML
# ---------------------------------------------------------------------------

HTML = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Labellisation pedigree</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #f0f2f5; color: #1a1a2e; }

    header {
      background: #1a1a2e; color: #fff; padding: 12px 24px;
      display: flex; align-items: center; justify-content: space-between;
    }
    header h1 { font-size: 1.1rem; font-weight: 600; }
    .badge {
      background: #e94560; color: #fff; border-radius: 12px;
      padding: 2px 10px; font-size: .85rem; font-weight: 700;
    }

    .layout { display: flex; height: calc(100vh - 50px); }

    /* Suivi par pays */
    .tracker { border-top: 1px solid #eee; padding-top: 10px; margin-top: 4px; }
    .tracker h2 { font-size: .9rem; color: #888; text-transform: uppercase;
                  letter-spacing: .05em; margin-bottom: 8px; }
    .country-row { margin-bottom: 6px; }
    .country-header {
      display: flex; justify-content: space-between; align-items: center;
      font-size: .82rem; font-weight: 700; color: #1a1a2e;
      cursor: pointer; user-select: none;
      padding: 4px 6px; border-radius: 4px;
    }
    .country-header:hover { background: #f0f2f5; }
    .country-header .flag { margin-right: 4px; }
    .country-total {
      background: #1a1a2e; color: #fff; border-radius: 10px;
      padding: 1px 8px; font-size: .75rem;
    }
    .book-list { padding-left: 12px; margin-top: 2px; display: none; }
    .book-list.open { display: block; }
    .book-row {
      display: flex; justify-content: space-between;
      font-size: .78rem; color: #555; padding: 2px 6px;
    }
    .book-count {
      background: #eef4ff; color: #4a90d9; border-radius: 8px;
      padding: 0 6px; font-weight: 600;
    }
    .bar-wrap { height: 6px; background: #eee; border-radius: 3px; margin-top: 3px; }
    .bar { height: 6px; border-radius: 3px; transition: width .4s; }
    .bar.ok       { background: #27ae60; }
    .bar.progress { background: #4a90d9; }
    .bar.low      { background: #e67e22; }
    .pct { font-size: .72rem; color: #888; margin-left: 4px; }
    .pct.ok { color: #27ae60; font-weight: 700; }

    /* Panneau gauche — document */
    .viewer {
      flex: 1; display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      background: #e8eaf0; padding: 16px; overflow: hidden;
    }
    .viewer img, .viewer iframe, .viewer embed {
      max-width: 100%; max-height: calc(100vh - 130px);
      border: 2px solid #ccc; border-radius: 4px; background: #fff;
      box-shadow: 0 4px 16px rgba(0,0,0,.15);
    }
    .viewer iframe, .viewer embed { width: 700px; height: calc(100vh - 130px); }
    .filename {
      margin-top: 8px; font-size: .78rem; color: #666;
      word-break: break-all; text-align: center;
    }

    /* Panneau droit — contrôles */
    .panel {
      width: 300px; background: #fff; display: flex; flex-direction: column;
      padding: 16px; gap: 10px; overflow-y: auto;
      border-left: 1px solid #dde;
    }
    .panel h2 { font-size: .9rem; color: #888; text-transform: uppercase;
                letter-spacing: .05em; margin-bottom: 4px; }

    .btn-class {
      display: flex; align-items: center; gap: 8px;
      width: 100%; padding: 10px 12px; border: 2px solid #dde;
      border-radius: 8px; background: #fafbff; cursor: pointer;
      font-size: .9rem; font-weight: 500; text-align: left;
      transition: all .15s;
    }
    .btn-class:hover { border-color: #4a90d9; background: #eef4ff; }
    .btn-class:active { transform: scale(.97); }
    .btn-class .key {
      background: #e8eaf0; border-radius: 4px; padding: 1px 6px;
      font-size: .75rem; font-weight: 700; color: #555; min-width: 20px;
      text-align: center;
    }
    .btn-class .count {
      margin-left: auto; background: #eef4ff; color: #4a90d9;
      border-radius: 10px; padding: 1px 8px; font-size: .78rem; font-weight: 700;
    }

    .separator { border: none; border-top: 1px solid #eee; margin: 4px 0; }

    .btn-reject {
      width: 100%; padding: 10px; border: 2px solid #f5c6cb;
      border-radius: 8px; background: #fff5f5; cursor: pointer;
      font-size: .9rem; font-weight: 600; color: #c0392b;
      transition: all .15s;
    }
    .btn-reject:hover { background: #fde8e8; border-color: #e74c3c; }

    .progress {
      font-size: .82rem; color: #888; text-align: center; margin-top: 4px;
    }
    .progress strong { color: #1a1a2e; }

    /* Toast */
    #toast {
      position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
      background: #27ae60; color: #fff; padding: 10px 24px;
      border-radius: 24px; font-weight: 600; font-size: .9rem;
      opacity: 0; transition: opacity .3s; pointer-events: none;
      box-shadow: 0 4px 12px rgba(0,0,0,.2);
    }

    .empty {
      flex: 1; display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 12px; color: #888;
    }
    .empty .icon { font-size: 3rem; }
  </style>
</head>
<body>

<header>
  <h1>Labellisation pedigree</h1>
  {% if filename %}
  <span class="badge">{{ remaining }} restant{{ 's' if remaining > 1 else '' }}</span>
  {% endif %}
</header>

<div class="layout">
  {% if filename %}

  <!-- Visualiseur : /preview renvoie toujours une image (JPEG) -->
  <div class="viewer">
    <img src="/preview/{{ filename }}" alt="{{ filename }}">
    <div class="filename">
      {{ filename }}
      {% if is_pdf %}<span style="color:#e94560;font-weight:600"> [PDF]</span>{% endif %}
    </div>
  </div>

  <!-- Panneau de classement -->
  <div class="panel">
    <h2>Classer dans</h2>

    {% for cls in classes %}
    <button class="btn-class" onclick="classify('{{ cls }}')" title="Touche {{ loop.index }}">
      <span class="key">{{ loop.index }}</span>
      {{ cls }}
      <span class="count">{{ stats[cls] }}</span>
    </button>
    {% endfor %}

    <hr class="separator">

    <button class="btn-reject" onclick="reject()" title="Touche R">
      ✕ &nbsp;Rejeter ce document
    </button>

    <div class="progress">
      <strong>{{ done }}</strong> classifiés &nbsp;·&nbsp;
      <strong>{{ remaining }}</strong> restants
    </div>

  </div>

  {% else %}
  <!-- Inbox vide : affiche le suivi dataset -->
  <div class="empty">
    <div class="icon">🎉</div>
    <p><strong>Inbox vide</strong> — tous les documents ont été traités.</p>
    <p style="font-size:.85rem; margin-bottom: 16px">Ajoutez des fichiers dans <code>data/inbox/</code> puis rechargez.</p>

    <div class="tracker" style="width:320px; background:#fff; border-radius:8px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,.08);">
      <h2 style="margin-bottom:4px">Suivi dataset</h2>
      <p style="font-size:.75rem; color:#888; margin-bottom:12px">Objectif : {{ target }} docs/livre pour lancer l'entraînement</p>
      {% for entry in country_stats %}
      <div class="country-row">
        <div class="country-header" onclick="toggleBooks(this)">
          <span>{{ entry.country }}</span>
          <span class="country-total">{{ entry.total }}</span>
        </div>
        <div class="book-list open">
          {% for book in entry.books %}
          {% set pct = [(book.count * 100 // target), 100] | min %}
          {% set bar_class = 'ok' if pct >= 100 else ('progress' if pct >= 40 else 'low') %}
          <div class="book-row">
            <span>{{ book.name }}</span>
            <span style="display:flex; align-items:center; gap:4px">
              <span class="book-count">{{ book.count }}</span>
              <span class="pct {{ 'ok' if pct >= 100 else '' }}">{{ pct }}%</span>
            </span>
          </div>
          <div class="bar-wrap">
            <div class="bar {{ bar_class }}" style="width: {{ pct }}%"></div>
          </div>
          {% endfor %}
        </div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}
</div>

<div id="toast"></div>

<script>
  const classes = {{ classes | tojson }};

  function post(url, body) {
    return fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    }).then(r => r.json());
  }

  function showToast(msg, color = '#27ae60') {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.style.background = color;
    t.style.opacity = '1';
    setTimeout(() => { t.style.opacity = '0'; }, 1200);
  }

  function classify(cls) {
    post('/classify', {class_name: cls})
      .then(d => {
        if (d.ok) { showToast('✓ ' + cls); setTimeout(() => location.reload(), 400); }
        else showToast(d.error, '#e74c3c');
      });
  }

  function reject() {
    post('/reject', {})
      .then(d => {
        if (d.ok) { showToast('✕ Rejeté', '#e74c3c'); setTimeout(() => location.reload(), 400); }
        else showToast(d.error, '#e74c3c');
      });
  }

  function toggleBooks(header) {
    const list = header.nextElementSibling;
    list.classList.toggle('open');
  }

  // Raccourcis clavier : 1-9 pour les classes, R pour rejeter
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    const n = parseInt(e.key);
    if (n >= 1 && n <= classes.length) { classify(classes[n - 1]); return; }
    if (e.key === 'r' || e.key === 'R') reject();
  });
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    classes = load_classes()
    files   = inbox_files()
    current = files[0] if files else None

    flat          = stats()
    country_stats = stats_by_country()
    return render_template_string(HTML,
        classes       = classes,
        stats         = flat,
        country_stats = country_stats,
        target        = MIN_DOCS_TARGET,
        filename      = current.name if current else None,
        is_pdf        = current.suffix.lower() == ".pdf" if current else False,
        remaining     = len(files),
        done          = sum(flat.values()),
    )


MIME_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".pdf":  "image/jpeg",  # converti en JPEG côté serveur
}

@app.route("/preview/<filename>")
def preview(filename: str):
    """Renvoie toujours une image (JPEG) quel que soit le format source."""
    path = INBOX / filename
    ext  = path.suffix.lower()

    if not path.exists() or ext not in ALLOWED_EXTENSIONS:
        return "Not found", 404

    if ext == ".pdf":
        data = pdf_first_page_jpeg(path)
        if data:
            return send_file(io.BytesIO(data), mimetype="image/jpeg")
        return "Conversion PDF échouée", 500

    return send_file(path, mimetype=MIME_MAP.get(ext, "image/jpeg"))


@app.route("/classify", methods=["POST"])
def classify():
    data       = request.get_json()
    class_name = data.get("class_name", "")
    classes    = load_classes()

    if class_name not in classes:
        return jsonify({"ok": False, "error": f"Classe inconnue : {class_name}"})

    files = inbox_files()
    if not files:
        return jsonify({"ok": False, "error": "Inbox vide."})

    move_to_class(files[0], class_name)
    return jsonify({"ok": True})


@app.route("/reject", methods=["POST"])
def reject():
    files = inbox_files()
    if not files:
        return jsonify({"ok": False, "error": "Inbox vide."})

    move_to_rejected(files[0])
    return jsonify({"ok": True})


@app.route("/stats")
def api_stats():
    return jsonify(stats())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    INBOX.mkdir(parents=True, exist_ok=True)
    REJECTED.mkdir(parents=True, exist_ok=True)
    for cls in load_classes():
        (RAW / cls).mkdir(parents=True, exist_ok=True)

    log.info("Interface de labellisation → http://localhost:5001")
    log.info("Déposez vos documents dans : %s", INBOX.resolve())
    app.run(host="0.0.0.0", port=5001, debug=False)
