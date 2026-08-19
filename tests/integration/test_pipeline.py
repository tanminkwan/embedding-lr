import pandas as pd
import pytest

from embedding_lr.constants import EMBEDDING_DIM
from embedding_lr.domain.models import Collection, Domain, KnowledgeItem, KnowledgeRecord, QueryRecord
from embedding_lr.embedding.pipeline import run
from embedding_lr.exceptions import DataValidationError


class FakeDataRepository:
    def __init__(self, records: list[QueryRecord]) -> None:
        self._records = records
        self.load_calls: list[str] = []

    def load(self, path: str) -> list[QueryRecord]:
        self.load_calls.append(path)
        return self._records

    def save(self, records: list[QueryRecord], path: str) -> None:
        raise NotImplementedError


class FakeVectorStore:
    """VectorStore Protocol의 fake 구현. upsert()는 등록된 레코드 수만큼 fake
    embedding(0.1로 채운 EMBEDDING_DIM 벡터)을 가진 KnowledgeItem을 만들어 저장한다."""

    def __init__(self) -> None:
        self.domains: list[Domain] = []
        self.collections: list[Collection] = []
        self._knowledge: dict[str, list[KnowledgeItem]] = {}
        self.upsert_calls: list[tuple[list[KnowledgeRecord], int, str]] = []
        self.get_knowledge_calls: list[tuple[int, str, int]] = []

    def list_domains(self) -> list[Domain]:
        return self.domains

    def create_domain(self, name: str) -> Domain:
        domain = Domain(id=len(self.domains) + 1, name=name)
        self.domains.append(domain)
        return domain

    def list_collections(self) -> list[Collection]:
        return self.collections

    def create_collection(self, name: str, collection_name: str) -> Collection:
        collection = Collection(name=name, collection_name=collection_name)
        self.collections.append(collection)
        return collection

    def upsert(self, records: list[KnowledgeRecord], domain_id: int, collection: str) -> None:
        self.upsert_calls.append((records, domain_id, collection))
        self._knowledge[collection] = [
            KnowledgeItem(
                id=str(i),
                collection=collection,
                content=record.content,
                extended_content=record.extended_content,
                domain_id=domain_id,
                source=record.source,
                created_at="2026-08-19T00:00:00Z",
                embedding=[0.1] * EMBEDDING_DIM,
            )
            for i, record in enumerate(records)
        ]

    def get_knowledge(self, domain_id: int, collection: str, limit: int) -> list[KnowledgeItem]:
        self.get_knowledge_calls.append((domain_id, collection, limit))
        return self._knowledge.get(collection, [])[:limit]


def _query_records(n: int) -> list[QueryRecord]:
    return [QueryRecord(query=f"q{i}", response=f"r{i}", category="IT") for i in range(n)]


class TestRun:
    def test_registers_and_writes_parquet_when_collection_empty(self, tmp_path):
        repo = FakeDataRepository(_query_records(2))
        store = FakeVectorStore()
        output_path = tmp_path / "train_vectors.parquet"

        run(repo, store, "data/v0.2/train.jsonl", str(output_path))

        assert len(store.upsert_calls) == 1
        registered_records, domain_id, collection = store.upsert_calls[0]
        assert len(registered_records) == 2
        assert collection == "v0_2_train"

        df = pd.read_parquet(output_path)
        assert len(df) == 2
        assert list(df.columns) == ["embedding", "label"]
        assert df["label"].tolist() == ["IT", "IT"]
        assert len(df["embedding"].iloc[0]) == EMBEDDING_DIM

    def test_skips_reregistration_when_count_matches(self, tmp_path):
        repo = FakeDataRepository(_query_records(2))
        store = FakeVectorStore()
        store.domains = [Domain(id=1, name="embedding_lr")]
        store.collections = [Collection(name="v0_2_train", collection_name="v0_2_train")]
        store._knowledge["v0_2_train"] = [
            KnowledgeItem(
                id=str(i),
                collection="v0_2_train",
                content=f"q{i}",
                extended_content=f"q{i}\nr{i}",
                domain_id=1,
                source="IT",
                created_at="2026-08-19T00:00:00Z",
                embedding=[0.9] * EMBEDDING_DIM,
            )
            for i in range(2)
        ]
        output_path = tmp_path / "train_vectors.parquet"

        run(repo, store, "data/v0.2/train.jsonl", str(output_path))

        assert store.upsert_calls == []
        df = pd.read_parquet(output_path)
        assert len(df) == 2
        assert df["embedding"].iloc[0][0] == pytest.approx(0.9)

    def test_raises_when_output_path_already_exists(self, tmp_path):
        repo = FakeDataRepository(_query_records(1))
        store = FakeVectorStore()
        output_path = tmp_path / "train_vectors.parquet"
        output_path.write_text("existing")

        with pytest.raises(DataValidationError, match="이미 존재"):
            run(repo, store, "data/v0.2/train.jsonl", str(output_path))

    def test_raises_when_reregistration_still_undercounts(self, tmp_path):
        repo = FakeDataRepository(_query_records(2))

        class AlwaysEmptyVectorStore(FakeVectorStore):
            def get_knowledge(self, domain_id: int, collection: str, limit: int) -> list[KnowledgeItem]:
                return []

        store = AlwaysEmptyVectorStore()
        output_path = tmp_path / "train_vectors.parquet"

        with pytest.raises(DataValidationError, match="재등록 후에도"):
            run(repo, store, "data/v0.2/train.jsonl", str(output_path))

        assert not output_path.exists()
