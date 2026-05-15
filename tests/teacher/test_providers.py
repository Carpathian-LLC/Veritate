# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tests for the provider registry surface.
# tests/teacher/test_providers.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import pytest

from veritate_mri.teacher.providers import (
    ENV_KEY,
    default_model_for,
    get_provider,
    list_providers,
    resolve_api_key,
)

# ------------------------------------------------------------------------------------
# Constants

# ------------------------------------------------------------------------------------
# Functions

def test_list_providers_order():
    """list_providers returns api kinds before local kinds, alphabetical within."""
    items = list_providers()
    kinds = [p["kind"] for p in items]
    api_count = kinds.count("api")
    assert kinds[:api_count] == ["api"] * api_count
    assert kinds[api_count:] == ["local"] * (len(items) - api_count)
    api_ids = [p["id"] for p in items[:api_count]]
    local_ids = [p["id"] for p in items[api_count:]]
    assert api_ids == sorted(api_ids)
    assert local_ids == sorted(local_ids)


def test_openai_auth_header():
    """get_provider openai has Authorization auth_header."""
    assert get_provider("openai")["auth_header"] == "Authorization"


def test_anthropic_auth_header():
    """get_provider anthropic has x-api-key auth_header."""
    assert get_provider("anthropic")["auth_header"] == "x-api-key"


def test_anthropic_system_style():
    """get_provider anthropic uses field system_message_style."""
    assert get_provider("anthropic")["system_message_style"] == "field"


def test_ollama_requires_key_false():
    """get_provider ollama has requires_key False."""
    assert get_provider("ollama")["requires_key"] is False


def test_unknown_provider_raises():
    """get_provider on missing id raises ValueError."""
    with pytest.raises(ValueError):
        get_provider("nope")


def test_resolve_api_key_env(monkeypatch):
    """resolve_api_key returns env value when set."""
    monkeypatch.setenv(ENV_KEY, "env_value")
    assert resolve_api_key("openai", "settings_key") == "env_value"


def test_resolve_api_key_settings(monkeypatch):
    """resolve_api_key falls back to settings when env unset."""
    monkeypatch.delenv(ENV_KEY, raising=False)
    assert resolve_api_key("openai", "settings_key") == "settings_key"


def test_resolve_api_key_none(monkeypatch):
    """resolve_api_key returns None when neither env nor settings set."""
    monkeypatch.delenv(ENV_KEY, raising=False)
    assert resolve_api_key("openai", None) is None


def test_resolve_api_key_ollama(monkeypatch):
    """resolve_api_key returns None for ollama regardless of inputs."""
    monkeypatch.setenv(ENV_KEY, "ignored")
    assert resolve_api_key("ollama", "ignored") is None


def test_default_model_openai():
    """default_model_for openai returns gpt-4o."""
    assert default_model_for("openai") == "gpt-4o"


def test_default_model_ollama_none():
    """default_model_for ollama returns None."""
    assert default_model_for("ollama") is None
