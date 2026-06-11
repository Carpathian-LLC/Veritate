# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - public surface for the teacher model module. provider-agnostic client used
#   for synthetic-corpus distillation. supports api providers (openai, anthropic,
#   etc.) and local servers (ollama, lm_studio, llama_cpp).
# - API key resolution: env var VERITATE_TEACHER_API_KEY first, falls back to settings teacher_api_key. Plaintext-at-rest in settings JSON (gitignored).
# veritate_mri/teacher/__init__.py
# ------------------------------------------------------------------------------------
# Imports:

from .client import Client, complete, TeacherError, TeacherAuthError, TeacherRateLimitError, TeacherUnavailableError
from .providers import list_providers, get_provider, resolve_api_key, default_model_for
from .test_connection import test, list_models

# ------------------------------------------------------------------------------------
# Constants

__all__ = [
    "Client",
    "complete",
    "TeacherError",
    "TeacherAuthError",
    "TeacherRateLimitError",
    "TeacherUnavailableError",
    "list_providers",
    "get_provider",
    "resolve_api_key",
    "default_model_for",
    "test",
    "list_models",
]

# ------------------------------------------------------------------------------------
# Functions
