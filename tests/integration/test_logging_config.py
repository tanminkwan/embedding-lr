import json
import logging

from embedding_lr.config import Settings
from embedding_lr.logging_config import setup_logging


def _settings(**overrides) -> Settings:
    defaults = dict(
        aipro_base_url="http://localhost:28000",
        aipro_api_token="test-token",
        embedding_server_base_url="http://localhost:8000",
        model_dir="models",
        model_path="models/model.pkl",
        service_name="embedding_lr",
        log_level="INFO",
    )
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def test_emits_json_line_with_required_schema_fields(capsys):
    setup_logging(_settings())
    logger = logging.getLogger("embedding_lr.embedding")

    logger.info("upsert 완료", extra={"phase": "embedding", "run_id": "20260819-000000-a1b2"})

    line = capsys.readouterr().out.strip()
    payload = json.loads(line)

    assert payload["level"] == "INFO"
    assert payload["service"] == "embedding_lr"
    assert payload["logger"] == "embedding_lr.embedding"
    assert payload["message"] == "upsert 완료"
    assert payload["phase"] == "embedding"
    assert payload["run_id"] == "20260819-000000-a1b2"
    assert "timestamp" in payload


def test_extra_context_field_is_nested_as_object(capsys):
    setup_logging(_settings())
    logger = logging.getLogger("embedding_lr.embedding")

    logger.info("처리 완료", extra={"extra": {"record_count": 128}})

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["extra"] == {"record_count": 128}


def test_log_level_filters_below_threshold(capsys):
    setup_logging(_settings(log_level="WARNING"))
    logger = logging.getLogger("embedding_lr.embedding")

    logger.info("이 메시지는 안 보여야 함")
    assert capsys.readouterr().out == ""

    logger.warning("이 메시지는 보여야 함")
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["level"] == "WARNING"
