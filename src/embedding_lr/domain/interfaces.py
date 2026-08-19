"""Protocol 정의 — P0_설계서_Common.md 4절. DIP 경계, 구현체는 각 Phase 모듈에서 제공."""

from typing import Protocol

from embedding_lr.domain.models import EmbeddingVector, QueryRecord


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def upsert(self, records: list[EmbeddingVector], collection: str) -> None:
        """category를 source 필드로 매핑해 AIPro+ POST /api/rag/knowledge 적재.
        콜렉션 전체를 재적재하는 방식이므로 레코드 단위 중복 판별은 하지 않는다."""
        ...


class Classifier(Protocol):
    def fit(self, X: list[list[float]], y: list[str]) -> None: ...

    def predict_proba(self, X: list[list[float]]) -> list[dict[str, float]]: ...


class DataRepository(Protocol):
    def load(self, path: str) -> list[QueryRecord]: ...

    def save(self, records: list[QueryRecord], path: str) -> None: ...
