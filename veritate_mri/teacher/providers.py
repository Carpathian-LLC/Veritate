# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - provider registry. one dict per provider keyed by id. cross-provider
#   differences (auth header, response shape, system message style) live as
#   data here; callers never branch on provider id outside this file.
# veritate_mri/teacher/providers.py
# ------------------------------------------------------------------------------------
# Imports:

import os

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_TIMEOUT_S = 60
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE_S = 1.0
DEFAULT_BACKOFF_MAX_S = 30.0
DEFAULT_MAX_CONCURRENCY = 16
# Raised from 4 on 2026-08-20. The old value protected a single small GPU from
# OOM, but it also capped corpus generation at roughly 36 days for a 100k
# conversation run on a 32-core / 275 GB box. The ceiling is now the user's
# choice; CONCURRENCY_CHOICES is what the dashboard offers.
LOCAL_MAX_CONCURRENCY = 256
MAX_CONCURRENCY = 256
CONCURRENCY_CHOICES = (2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
ENV_KEY = "VERITATE_TEACHER_API_KEY"
RETRY_STATUS = (408, 429, 500, 502, 503, 504)

OPENAI_RESPONSE_PATH = ["choices", 0, "message", "content"]
ANTHROPIC_RESPONSE_PATH = ["content", 0, "text"]

# Optional per-provider capability keys. Omitted from an entry means the
# default below; the client never infers a capability from a URL string.
# - supports_stream: emits OpenAI-style `data: {...delta...}` chunks, so a
#   cancellable streaming request can abort mid-generation.
# - unload_path / unload_body: native "drop this model from memory" endpoint.
DEFAULT_SUPPORTS_STREAM = True
DEFAULT_UNLOAD_PATH     = ""
DEFAULT_UNLOAD_BODY     = {}
OLLAMA_UNLOAD_PATH = "/api/generate"
OLLAMA_UNLOAD_BODY = {"keep_alive": 0}

PROVIDERS = {
    "carpathian": {
        "id": "carpathian",
        "display_name": "Carpathian AI",
        "kind": "api",
        "base_url": "https://api.carpathian.ai/ai/v1",
        "chat_path": "/chat/completions",
        "models_path": "/models",
        "auth_style": "bearer",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "extra_headers": {},
        "default_models": ["default"],
        "response_text_path": list(OPENAI_RESPONSE_PATH),
        "messages_key": "messages",
        "system_message_style": "inline",
        "requires_key": True,
        "model_selectable": False,
    },
    "openai": {
        "id": "openai",
        "display_name": "OpenAI",
        "kind": "api",
        "base_url": "https://api.openai.com",
        "chat_path": "/v1/chat/completions",
        "models_path": "/v1/models",
        "auth_style": "bearer",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "extra_headers": {},
        "default_models": ["gpt-4o", "gpt-4o-mini", "o1-mini"],
        "response_text_path": list(OPENAI_RESPONSE_PATH),
        "messages_key": "messages",
        "system_message_style": "inline",
        "requires_key": True,
    },
    "anthropic": {
        "id": "anthropic",
        "display_name": "Anthropic",
        "kind": "api",
        "base_url": "https://api.anthropic.com",
        "chat_path": "/v1/messages",
        "models_path": "/v1/models",
        "auth_style": "x-api-key",
        "auth_header": "x-api-key",
        "auth_prefix": "",
        "extra_headers": {"anthropic-version": "2023-06-01"},
        "default_models": ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        "response_text_path": list(ANTHROPIC_RESPONSE_PATH),
        "messages_key": "messages",
        "system_message_style": "field",
        "requires_key": True,
        "supports_stream": False,
    },
    "gemini": {
        "id": "gemini",
        "display_name": "Google Gemini",
        "kind": "api",
        "base_url": "https://generativelanguage.googleapis.com",
        "chat_path": "/v1beta/openai/chat/completions",
        "models_path": "/v1beta/openai/models",
        "auth_style": "bearer",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "extra_headers": {},
        "default_models": ["gemini-2.0-flash", "gemini-2.5-pro"],
        "response_text_path": list(OPENAI_RESPONSE_PATH),
        "messages_key": "messages",
        "system_message_style": "inline",
        "requires_key": True,
    },
    "xai": {
        "id": "xai",
        "display_name": "xAI Grok",
        "kind": "api",
        "base_url": "https://api.x.ai",
        "chat_path": "/v1/chat/completions",
        "models_path": "/v1/models",
        "auth_style": "bearer",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "extra_headers": {},
        "default_models": ["grok-2-latest", "grok-2-mini-latest"],
        "response_text_path": list(OPENAI_RESPONSE_PATH),
        "messages_key": "messages",
        "system_message_style": "inline",
        "requires_key": True,
    },
    "deepseek": {
        "id": "deepseek",
        "display_name": "DeepSeek",
        "kind": "api",
        "base_url": "https://api.deepseek.com",
        "chat_path": "/v1/chat/completions",
        "models_path": "/v1/models",
        "auth_style": "bearer",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "extra_headers": {},
        "default_models": ["deepseek-chat", "deepseek-reasoner"],
        "response_text_path": list(OPENAI_RESPONSE_PATH),
        "messages_key": "messages",
        "system_message_style": "inline",
        "requires_key": True,
    },
    "mistral": {
        "id": "mistral",
        "display_name": "Mistral",
        "kind": "api",
        "base_url": "https://api.mistral.ai",
        "chat_path": "/v1/chat/completions",
        "models_path": "/v1/models",
        "auth_style": "bearer",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "extra_headers": {},
        "default_models": ["mistral-large-latest", "mistral-small-latest"],
        "response_text_path": list(OPENAI_RESPONSE_PATH),
        "messages_key": "messages",
        "system_message_style": "inline",
        "requires_key": True,
    },
    "groq": {
        "id": "groq",
        "display_name": "Groq",
        "kind": "api",
        "base_url": "https://api.groq.com",
        "chat_path": "/openai/v1/chat/completions",
        "models_path": "/openai/v1/models",
        "auth_style": "bearer",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "extra_headers": {},
        "default_models": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        "response_text_path": list(OPENAI_RESPONSE_PATH),
        "messages_key": "messages",
        "system_message_style": "inline",
        "requires_key": True,
    },
    "openrouter": {
        "id": "openrouter",
        "display_name": "OpenRouter",
        "kind": "api",
        "base_url": "https://openrouter.ai",
        "chat_path": "/api/v1/chat/completions",
        "models_path": "/api/v1/models",
        "auth_style": "bearer",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "extra_headers": {},
        "default_models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
        "response_text_path": list(OPENAI_RESPONSE_PATH),
        "messages_key": "messages",
        "system_message_style": "inline",
        "requires_key": True,
    },
    "ollama": {
        "id": "ollama",
        "display_name": "Ollama",
        "kind": "local",
        "base_url": "http://localhost:11434",
        "chat_path": "/v1/chat/completions",
        "models_path": "/api/tags",
        "models_array": "models",
        "models_id": "name",
        "unload_path": OLLAMA_UNLOAD_PATH,
        "unload_body": dict(OLLAMA_UNLOAD_BODY),
        "auth_style": "none",
        "auth_header": "",
        "auth_prefix": "",
        "extra_headers": {},
        "default_models": [],
        "response_text_path": list(OPENAI_RESPONSE_PATH),
        "messages_key": "messages",
        "system_message_style": "inline",
        "requires_key": False,
    },
    "lm_studio": {
        "id": "lm_studio",
        "display_name": "LM Studio",
        "kind": "local",
        "base_url": "http://localhost:1234",
        "chat_path": "/v1/chat/completions",
        "models_path": "/v1/models",
        "auth_style": "none",
        "auth_header": "",
        "auth_prefix": "",
        "extra_headers": {},
        "default_models": [],
        "response_text_path": list(OPENAI_RESPONSE_PATH),
        "messages_key": "messages",
        "system_message_style": "inline",
        "requires_key": False,
    },
    "llama_cpp": {
        "id": "llama_cpp",
        "display_name": "llama.cpp",
        "kind": "local",
        "base_url": "http://localhost:8080",
        "chat_path": "/v1/chat/completions",
        "models_path": "/v1/models",
        "auth_style": "none",
        "auth_header": "",
        "auth_prefix": "",
        "extra_headers": {},
        "default_models": [],
        "response_text_path": list(OPENAI_RESPONSE_PATH),
        "messages_key": "messages",
        "system_message_style": "inline",
        "requires_key": False,
    },
}

# ------------------------------------------------------------------------------------
# Functions

def _copy(p):
    out = dict(p)
    out["extra_headers"] = dict(p["extra_headers"])
    out["default_models"] = list(p["default_models"])
    out["response_text_path"] = list(p["response_text_path"])
    if "unload_body" in p:
        out["unload_body"] = dict(p["unload_body"])
    return out


def list_providers():
    items = [_copy(p) for p in PROVIDERS.values()]
    items.sort(key=lambda p: (0 if p["kind"] == "api" else 1, p["id"]))
    return items


def get_provider(provider_id):
    p = PROVIDERS.get(provider_id)
    if p is None:
        raise ValueError(f"unknown provider: {provider_id}")
    return _copy(p)


def resolve_api_key(provider_id, settings_key):
    p = PROVIDERS.get(provider_id)
    if p is None:
        raise ValueError(f"unknown provider: {provider_id}")
    if not p["requires_key"]:
        return None
    env = os.environ.get(ENV_KEY)
    if env:
        return env
    if settings_key:
        return settings_key
    return None


def default_model_for(provider_id):
    p = PROVIDERS.get(provider_id)
    if p is None:
        raise ValueError(f"unknown provider: {provider_id}")
    if p["default_models"]:
        return p["default_models"][0]
    return None
