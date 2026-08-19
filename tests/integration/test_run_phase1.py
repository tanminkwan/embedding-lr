import json

import pytest

from embedding_lr.cli.run_phase1 import main
from embedding_lr.data_generation.jsonl_repository import JsonlRepository


REQUIRED_ENV = {
    "AIPRO_BASE_URL": "http://localhost:28000",
    "AIPRO_API_TOKEN": "test-token",
    "EMBEDDING_SERVER_BASE_URL": "http://localhost:8000",
    "MODEL_DIR": "models",
}


def _set_env(monkeypatch, tmp_path):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("STATUS_DIR", str(tmp_path / "status"))


class TestRunPhase1:
    def test_converts_csv_to_jsonl(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_path = tmp_path / "role_01_middleware.csv"
        output_path = tmp_path / "role_01_middleware.jsonl"
        input_path.write_text(
            "질의,응답,카테고리\nnginx 재시작?,systemctl restart nginx,IT\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "sys.argv",
            ["run_phase1", "--input", str(input_path), "--output", str(output_path)],
        )

        main()

        records = JsonlRepository().load(str(output_path))
        assert len(records) == 1
        assert records[0].query == "nginx 재시작?"
        assert records[0].category == "IT"

    def test_records_succeeded_status(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_path = tmp_path / "role_01_middleware.csv"
        output_path = tmp_path / "role_01_middleware.jsonl"
        input_path.write_text("질의,응답,카테고리\nq,r,IT\n", encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            ["run_phase1", "--input", str(input_path), "--output", str(output_path)],
        )

        main()

        status_files = list((tmp_path / "status").glob("phase1_*.json"))
        assert len(status_files) == 1
        status = json.loads(status_files[0].read_text(encoding="utf-8"))
        assert status["status"] == "succeeded"

    def test_raises_when_output_already_exists(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_path = tmp_path / "role_01_middleware.csv"
        output_path = tmp_path / "role_01_middleware.jsonl"
        input_path.write_text("질의,응답,카테고리\nq,r,IT\n", encoding="utf-8")
        output_path.write_text("기존 내용\n", encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            ["run_phase1", "--input", str(input_path), "--output", str(output_path)],
        )

        with pytest.raises(Exception, match="이미 존재"):
            main()
