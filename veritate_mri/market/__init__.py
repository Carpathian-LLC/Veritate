# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Market-prediction package: data layer (data.py), byte-model serving (veritate.py),
#   live feed (live.py), and a corpus-manifest CLI (corpus_manifest.py). The web server
#   is shared. The byte model is the only engine; the old GBDT baseline was removed.
# - The byte-model path (veritate.py) reads the canonical Veritate model registry and
#   checkpoints READ-ONLY to run trained models; it never mutates canonical
#   training/chat/RAG state or writes into their directories.
# veritate_mri/market/__init__.py
# ------------------------------------------------------------------------------------
