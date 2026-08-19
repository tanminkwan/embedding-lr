import pytest
from pydantic import ValidationError

from embedding_lr.config import Settings


REQUIRED_ENV = {
    "AIPRO_BASE_URL": "http://localhost:28000",
    "AIPRO_API_TOKEN": "test-token",
    "MODEL_DIR": "models",
}


def test_loads_required_fields_from_env(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)

    settings = Settings(_env_file=None)

    assert settings.aipro_base_url == "http://localhost:28000"
    assert settings.aipro_api_token == "test-token"
    assert settings.model_dir == "models"


def test_applies_defaults_when_optional_fields_absent(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)

    settings = Settings(_env_file=None)

    assert settings.log_level == "INFO"
    assert settings.service_name == "embedding_lr"
    assert settings.env == "local"
    assert settings.status_dir == "status"
    assert settings.aipro_timeout_seconds == 30.0


def test_overrides_defaults_from_env(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"


def test_missing_required_field_raises(monkeypatch):
    monkeypatch.delenv("AIPRO_BASE_URL", raising=False)
    monkeypatch.delenv("AIPRO_API_TOKEN", raising=False)
    monkeypatch.delenv("MODEL_DIR", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
