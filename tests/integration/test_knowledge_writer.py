import pytest

from embedding_lr.domain.models import Collection, Domain, KnowledgeItem, KnowledgeRecord, QueryRecord
from embedding_lr.embedding.knowledge_writer import write_knowledge
from embedding_lr.exceptions import DataValidationError


class FakeVectorStore:
    """VectorStore Protocol의 fake 구현 — HTTP 없이 upsert() 호출 내용만 검증."""

    def __init__(self) -> None:
        self.upsert_calls: list[tuple[list[KnowledgeRecord], int, str]] = []

    def list_domains(self) -> list[Domain]:
        raise NotImplementedError

    def create_domain(self, name: str) -> Domain:
        raise NotImplementedError

    def list_collections(self) -> list[Collection]:
        raise NotImplementedError

    def create_collection(self, name: str, collection_name: str) -> Collection:
        raise NotImplementedError

    def upsert(self, records: list[KnowledgeRecord], domain_id: int, collection: str) -> None:
        self.upsert_calls.append((records, domain_id, collection))

    def get_knowledge(self, domain_id: int, collection: str, limit: int) -> list[KnowledgeItem]:
        raise NotImplementedError


class TestWriteKnowledge:
    def test_maps_query_to_content_and_category_to_source(self):
        store = FakeVectorStore()
        records = [QueryRecord(query="q1", response="r1", category="IT")]

        write_knowledge(store, records, domain_id=1, collection="v0_2_train")

        [(knowledge_records, domain_id, collection)] = store.upsert_calls
        assert domain_id == 1
        assert collection == "v0_2_train"
        assert knowledge_records == [
            KnowledgeRecord(content="q1", extended_content="q1\nr1", source="IT")
        ]

    def test_upserts_all_records_in_a_single_call(self):
        store = FakeVectorStore()
        records = [
            QueryRecord(query="q1", response="r1", category="IT"),
            QueryRecord(query="q2", response="r2", category="DAILY"),
        ]

        write_knowledge(store, records, domain_id=1, collection="v0_2_train")

        assert len(store.upsert_calls) == 1
        knowledge_records, _, _ = store.upsert_calls[0]
        assert len(knowledge_records) == 2

    def test_raises_data_validation_error_when_category_is_missing(self):
        store = FakeVectorStore()
        records = [QueryRecord(query="q1", response="r1", category=None)]

        with pytest.raises(DataValidationError, match="레코드 0"):
            write_knowledge(store, records, domain_id=1, collection="v0_2_train")

        assert store.upsert_calls == []
