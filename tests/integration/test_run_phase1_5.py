import json

from embedding_lr.cli.run_phase1_5 import main
from embedding_lr.data_generation.jsonl_repository import JsonlRepository
from embedding_lr.domain.models import QueryRecord


REQUIRED_ENV = {
    "AIPRO_BASE_URL": "http://localhost:28000",
    "AIPRO_API_TOKEN": "test-token",
    "EMBEDDING_SERVER_BASE_URL": "http://localhost:8000",
    "MODEL_DIR": "models",
}

_IT_ROLE_FILES = [
    "role_01_middleware",
    "role_02_os",
    "role_03_network",
    "role_04_dba",
    "role_05_devops",
]
_NON_IT_ROLE_FILES = {
    "role_06_daily": "DAILY",
    "role_07_knowledge": "KNOWLEDGE",
    "role_08_creative": "CREATIVE",
    "role_09_anomaly": "ANOMALY",
}


def _set_env(monkeypatch, tmp_path):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("STATUS_DIR", str(tmp_path / "status"))


def _write_role_files(input_dir):
    repo = JsonlRepository()
    input_dir.mkdir(parents=True, exist_ok=True)

    for i, role_file in enumerate(_IT_ROLE_FILES):
        records = [
            QueryRecord(query=f"{role_file}-q{n}", response=f"r{n}", category="IT")
            for n in range(40)
        ]
        repo.save(records, str(input_dir / f"{role_file}.jsonl"))

    for role_file, category in _NON_IT_ROLE_FILES.items():
        records = [
            QueryRecord(query=f"{role_file}-q{n}", response=f"r{n}", category=category)
            for n in range(200)
        ]
        repo.save(records, str(input_dir / f"{role_file}.jsonl"))


class TestRunPhase1_5:
    def test_combines_and_splits_role_files(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _write_role_files(input_dir)

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_phase1_5",
                "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
            ],
        )

        main()

        repo = JsonlRepository()
        data = repo.load(str(output_dir / "data.jsonl"))
        train = repo.load(str(output_dir / "train.jsonl"))
        test = repo.load(str(output_dir / "test.jsonl"))
        val = repo.load(str(output_dir / "val.jsonl"))

        assert len(data) == 1000
        assert len(train) == 600
        assert len(test) == 200
        assert len(val) == 200

    def test_records_succeeded_status(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _write_role_files(input_dir)

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_phase1_5",
                "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
            ],
        )

        main()

        status_files = list((tmp_path / "status").glob("phase1_5_*.json"))
        assert len(status_files) == 1
        status = json.loads(status_files[0].read_text(encoding="utf-8"))
        assert status["status"] == "succeeded"
