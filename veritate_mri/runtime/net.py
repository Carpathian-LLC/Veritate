# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - shared HTTPS trust store. certifi CA bundle when importable, else system default.
#   framework Python on macOS ships a broken store unless the user runs
#   "Install Certificates.command"; certifi sidesteps that.
# veritate_mri/runtime/net.py
# ------------------------------------------------------------------------------------
# Imports:

import ssl

# ------------------------------------------------------------------------------------
# Constants

# ------------------------------------------------------------------------------------
# Functions

def ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()
