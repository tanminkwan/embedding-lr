import json

import httpx
import pandas as pd
import respx

from embedding_lr.cli.run_phase2 import main
from embedding_lr.constants import EMBEDDING_DIM
from embedding_lr.data_generation.jsonl_repository import JsonlRepository
from embedding_lr.domain.models import QueryRecord

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


def _write_input_jsonl(path, n=2):
    repo = JsonlRepository()
    records = [QueryRecord(query=f"q{i}", response=f"r{i}", category="IT") for i in range(n)]
    repo.save(records, str(path))


def _knowledge_item(i, collection):
    return {
        "id": str(i),
        "collection": collection,
        "content": f"q{i}",
        "extended_content": f"q{i}\nr{i}",
        "domain_id": 1,
        "source": "IT",
        "created_at": "2026-08-19T00:00:00Z",
        "embedding": [0.1] * EMBEDDING_DIM,
    }


def _mock_aipro(*, existing_count_before_registration: int, record_count: int, collection: str):
    respx.get("http://localhost:28000/api/domains").mock(return_value=httpx.Response(200, json=[]))
    respx.post("http://localhost:28000/api/domains").mock(
        return_value=httpx.Response(200, json={"id": 1, "name": "embedding_lr"})
    )
    respx.get("http://localhost:28000/api/collections").mock(return_value=httpx.Response(200, json=[]))
    respx.post("http://localhost:28000/api/collections").mock(
        return_value=httpx.Response(200, json={"name": collection, "collection_name": collection})
    )
    respx.post("http://localhost:28000/api/rag/knowledge").mock(return_value=httpx.Response(200, json={}))
    respx.get("http://localhost:28000/api/rag/knowledge").mock(
        side_effect=[
            httpx.Response(
                200,
                json=[_knowledge_item(i, collection) for i in range(existing_count_before_registration)],
            ),
            httpx.Response(
                200, json=[_knowledge_item(i, collection) for i in range(record_count)]
            ),
        ]
    )


class TestRunPhase2:
    @respx.mock
    def test_registers_and_writes_vectors_parquet(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_path = tmp_path / "data" / "v0.2" / "train.jsonl"
        input_path.parent.mkdir(parents=True)
        _write_input_jsonl(input_path, n=2)
        output_path = tmp_path / "train_vectors.parquet"
        _mock_aipro(existing_count_before_registration=0, record_count=2, collection="v0_2_train")

        monkeypatch.setattr(
            "sys.argv",
            ["run_phase2", "--input", str(input_path), "--output", str(output_path)],
        )

        main()

        df = pd.read_parquet(output_path)
        assert len(df) == 2
        assert df["label"].tolist() == ["IT", "IT"]
        assert len(df["embedding"].iloc[0]) == EMBEDDING_DIM

    @respx.mock
    def test_records_succeeded_status(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_path = tmp_path / "data" / "v0.2" / "train.jsonl"
        input_path.parent.mkdir(parents=True)
        _write_input_jsonl(input_path, n=2)
        output_path = tmp_path / "train_vectors.parquet"
        _mock_aipro(existing_count_before_registration=0, record_count=2, collection="v0_2_train")

        monkeypatch.setattr(
            "sys.argv",
            ["run_phase2", "--input", str(input_path), "--output", str(output_path)],
        )

        main()

        status_files = list((tmp_path / "status").glob("phase2_*.json"))
        assert len(status_files) == 1
        status = json.loads(status_files[0].read_text(encoding="utf-8"))
        assert status["status"] == "succeeded"
