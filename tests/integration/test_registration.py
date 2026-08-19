from embedding_lr.domain.models import Collection, Domain, KnowledgeItem, KnowledgeRecord
from embedding_lr.embedding.registration import ensure_collection, ensure_domain


class FakeVectorStore:
    """VectorStore Protocol의 fake 구현 — HTTP 없이 registration.py의 idempotent 판단 로직만 검증."""

    def __init__(self, domains: list[Domain] | None = None, collections: list[Collection] | None = None) -> None:
        self.domains = list(domains or [])
        self.collections = list(collections or [])
        self.create_domain_calls: list[str] = []
        self.create_collection_calls: list[tuple[str, str]] = []

    def list_domains(self) -> list[Domain]:
        return self.domains

    def create_domain(self, name: str) -> Domain:
        self.create_domain_calls.append(name)
        domain = Domain(id=len(self.domains) + 1, name=name)
        self.domains.append(domain)
        return domain

    def list_collections(self) -> list[Collection]:
        return self.collections

    def create_collection(self, name: str, collection_name: str) -> Collection:
        self.create_collection_calls.append((name, collection_name))
        collection = Collection(name=name, collection_name=collection_name)
        self.collections.append(collection)
        return collection

    def upsert(self, records: list[KnowledgeRecord], domain_id: int, collection: str) -> None:
        raise NotImplementedError

    def get_knowledge(self, domain_id: int, collection: str, limit: int) -> list[KnowledgeItem]:
        raise NotImplementedError


class TestEnsureDomain:
    def test_returns_existing_domain_without_creating(self):
        existing = Domain(id=1, name="embedding_lr")
        store = FakeVectorStore(domains=[existing])

        result = ensure_domain(store, "embedding_lr")

        assert result == existing
        assert store.create_domain_calls == []

    def test_creates_domain_when_not_found(self):
        store = FakeVectorStore(domains=[])

        result = ensure_domain(store, "embedding_lr")

        assert result.name == "embedding_lr"
        assert store.create_domain_calls == ["embedding_lr"]


class TestEnsureCollection:
    def test_returns_existing_collection_without_creating(self):
        existing = Collection(name="v0_2_train", collection_name="v0_2_train")
        store = FakeVectorStore(collections=[existing])

        result = ensure_collection(store, "v0_2_train", "v0_2_train")

        assert result == existing
        assert store.create_collection_calls == []

    def test_creates_collection_when_not_found(self):
        store = FakeVectorStore(collections=[])

        result = ensure_collection(store, "v0_2_train", "v0_2_train")

        assert result.collection_name == "v0_2_train"
        assert store.create_collection_calls == [("v0_2_train", "v0_2_train")]
