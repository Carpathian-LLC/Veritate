# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - in-app wiki: categories, per-category entry lists, single entries.
# veritate_mri/routes/wiki_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from readers import wiki as wiki_reader

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/wiki")
    def wiki_index():
        return {"categories": wiki_reader.list_categories()}

    @app.route("/wiki/<category>")
    def wiki_category(category):
        entries = wiki_reader.list_entries(category)
        if entries is None:
            return ({"error": f"category not found: {category}"}, 404)
        return {"category": category, "entries": entries}

    @app.route("/wiki/<category>/<slug>")
    def wiki_entry(category, slug):
        entry = wiki_reader.load_entry(category, slug)
        if entry is None:
            return ({"error": f"entry not found: {category}/{slug}"}, 404)
        return entry
