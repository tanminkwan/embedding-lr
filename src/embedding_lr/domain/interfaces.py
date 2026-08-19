"""Protocol 정의 — P0_설계서_Common.md 4절. DIP 경계, 구현체는 각 Phase 모듈에서 제공."""

from typing import Protocol

from embedding_lr.domain.models import Collection, Domain, KnowledgeItem, KnowledgeRecord, QueryRecord


class EmbeddingClient(Protocol):
    """독립 Embedding Service(localhost:8000, AIPro+와 무관) 호출 — Phase 5 추론 전용.
    Phase 2는 이 클라이언트를 쓰지 않는다(AIPro+ 일괄 등록/조회로 대체, VectorStore 참고)."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    """AIPro+(localhost:28000) 지식 데이터 저장소 호출 — Phase 2(학습) 전용."""

    def list_domains(self) -> list[Domain]: ...

    def create_domain(self, name: str) -> Domain: ...

    def list_collections(self) -> list[Collection]: ...

    def create_collection(self, name: str, collection_name: str) -> Collection: ...

    def upsert(self, records: list[KnowledgeRecord], domain_id: int, collection: str) -> None:
        """AIPro+ POST /api/rag/knowledge로 레코드 1건씩 개별 등록(content 기반 —
        AIPro+가 내부에서 임베딩을 계산해 저장). 벌크 업로드 엔드포인트(`/api/rag/bulk-upload`)는
        쓰지 않는다(사용자 확인, 2026-08-19). 콜렉션 전체를 재적재하는 방식이므로 레코드
        단위 중복 판별은 하지 않는다."""
        ...

    def get_knowledge(self, domain_id: int, collection: str, limit: int) -> list[KnowledgeItem]:
        """AIPro+ GET /api/rag/knowledge — 임베딩 포함 조회. upsert()로 등록된 데이터를
        일괄 조회해 embedding/pipeline.py가 *_vectors.parquet을 만드는 데 사용한다."""
        ...


class Classifier(Protocol):
    def fit(self, X: list[list[float]], y: list[str]) -> None: ...

    def predict_proba(self, X: list[list[float]]) -> list[dict[str, float]]: ...


class DataRepository(Protocol):
    def load(self, path: str) -> list[QueryRecord]: ...

    def save(self, records: list[QueryRecord], path: str) -> None: ...
