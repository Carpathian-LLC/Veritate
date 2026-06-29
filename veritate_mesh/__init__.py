# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate mesh: peer mesh of veritate nodes with one elected coordinator (hub).
# - one binary, two roles. role selected in runtime/settings.py (off/node/hub/both).
# - transport: http over user-provided network (tailscale recommended). pull-based
#   job dispatch so nodes need no inbound port.
# - mesh never crosses the per-byte decode boundary. only training / data-gen /
#   eval cross box boundaries.
# veritate_mesh/__init__.py
# ------------------------------------------------------------------------------------
