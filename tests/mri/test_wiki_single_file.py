# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - route tests for the single-file wiki (/wiki, /wiki/doc, /wiki/<slug>/page).
#   Reads the real repo-root documentation.md: it is checked-in and stable, so the
#   assertions are deterministic without stubbing the reader (rule 33).
# tests/mri/test_wiki_single_file.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import Flask
from routes import wiki_routes

# ------------------------------------------------------------------------------------
# Constants

SETTING_SLUGS = [
    "recipe", "optimizer", "trunk", "precision", "batch_size", "seq", "n_chunks",
    "bptt_window", "base_lr", "min_lr", "lr_schedule", "warmup_steps", "weight_decay",
    "grad_clip", "label_smoothing",
]

# ------------------------------------------------------------------------------------
# Functions

def _client():
    """Minimal app with wiki_routes registered against the real documentation.md."""
    app = Flask(__name__)
    wiki_routes.register(app)
    return app.test_client()


def test_toc_contains_every_training_form_settings_slug():
    """GET /wiki lists a section for each slug the training form's learn-more links use."""
    slugs = {s["slug"] for s in wiki_routes.wiki_reader.toc()}
    assert set(SETTING_SLUGS) <= slugs


def test_wiki_index_returns_sections():
    """GET /wiki returns a non-empty sections list."""
    body = _client().get("/wiki").get_json()
    assert body["sections"]


def test_wiki_doc_body_html_contains_veritate():
    """GET /wiki/doc renders the whole document and mentions the platform name."""
    body = _client().get("/wiki/doc").get_json()
    assert body["body_html"] and "veritate" in body["body_html"].lower()


def test_settings_page_returns_learning_rate_content():
    """GET /wiki/settings/base_lr/page renders the base_lr section for the learn-more link."""
    r = _client().get("/wiki/settings/base_lr/page")
    assert r.status_code == 200
    assert "learning rate" in r.get_data(as_text=True).lower()


def test_unknown_slug_page_returns_404():
    """GET /wiki/<unknown>/page returns 404 rather than an empty page."""
    r = _client().get("/wiki/nonexistent_slug_xyz/page")
    assert r.status_code == 404
