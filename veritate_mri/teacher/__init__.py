# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - public surface for the teacher model module. provider-agnostic client used
#   for synthetic-corpus distillation. supports api providers (openai, anthropic,
#   etc.) and local servers (ollama, lm_studio, llama_cpp).
# - API key resolution: env var VERITATE_TEACHER_API_KEY first, falls back to settings
#   teacher_api_key. Plaintext-at-rest in settings JSON (gitignored).
# veritate_mri/teacher/__init__.py
# ------------------------------------------------------------------------------------
# Imports:

from .client import Client, TeacherAuthError, TeacherError, TeacherRateLimitError, TeacherUnavailableError, complete
from .providers import default_model_for, get_provider, list_providers, resolve_api_key
from .test_connection import list_models, test

# ------------------------------------------------------------------------------------
# Constants

__all__ = [
    "Client",
    "TeacherAuthError",
    "TeacherError",
    "TeacherRateLimitError",
    "TeacherUnavailableError",
    "complete",
    "default_model_for",
    "get_provider",
    "list_models",
    "list_providers",
    "resolve_api_key",
    "test",
]

# ------------------------------------------------------------------------------------
# Functions
