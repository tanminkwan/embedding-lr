import json
import re

import pytest

from embedding_lr.config import Settings
from embedding_lr.workflow.run_context import new_run_id, run_context

RUN_ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        aipro_base_url="http://localhost:28000",
        aipro_api_token="test-token",
        model_dir=str(tmp_path / "models"),
        status_dir=str(tmp_path / "status"),
    )


def test_new_run_id_matches_expected_format():
    assert RUN_ID_PATTERN.match(new_run_id())


def test_success_path_writes_succeeded_status(tmp_path):
    settings = _settings(tmp_path)

    with run_context("phase2", settings) as (run_id, logger):
        logger.info("작업 수행")

    status_path = tmp_path / "status" / f"phase2_{run_id}.json"
    payload = json.loads(status_path.read_text())

    assert payload["run_id"] == run_id
    assert payload["phase"] == "phase2"
    assert payload["status"] == "succeeded"
    assert payload["started_at"] is not None
    assert payload["ended_at"] is not None
    assert payload["error"] is None


def test_failure_path_writes_failed_status_and_reraises(tmp_path):
    settings = _settings(tmp_path)
    captured_run_id = None

    with pytest.raises(RuntimeError, match="boom"):
        with run_context("phase2", settings) as (run_id, _logger):
            captured_run_id = run_id
            raise RuntimeError("boom")

    status_path = tmp_path / "status" / f"phase2_{captured_run_id}.json"
    payload = json.loads(status_path.read_text())

    assert payload["status"] == "failed"
    assert payload["error"] == "boom"
    assert payload["ended_at"] is not None
