"""QueryRecord(정제 완료) → KnowledgeRecord 매핑 후 VectorStore.upsert() 적재 —
Architecture_Design.md 2절/4절. 등급 B(오케스트레이션) — VectorStore Protocol에만
의존(DIP), fake로 통합 테스트. text_cleaner 호출과 query+response 결합은 이 모듈이 아니라
호출부(embedding/pipeline.py)의 책임이다(P2_설계서_TextCleaning.md 2절 SRP)."""

from pydantic import ValidationError as PydanticValidationError

from embedding_lr.domain.interfaces import VectorStore
from embedding_lr.domain.models import KnowledgeRecord, QueryRecord
from embedding_lr.exceptions import DataValidationError


def write_knowledge(store: VectorStore, records: list[QueryRecord], domain_id: int, collection: str) -> None:
    """`records`(정제된 query/response, category 필수) 전체를 `collection`에 적재한다.
    콜렉션 전체 재등록 방식이므로 레코드 단위 중복 판별은 하지 않는다."""
    knowledge_records = [_to_knowledge_record(record, index) for index, record in enumerate(records)]
    store.upsert(knowledge_records, domain_id=domain_id, collection=collection)


def _to_knowledge_record(record: QueryRecord, index: int) -> KnowledgeRecord:
    try:
        return KnowledgeRecord(
            content=record.query,
            extended_content=f"{record.query}\n{record.response}",
            source=record.category,
        )
    except PydanticValidationError as exc:
        raise DataValidationError(f"레코드 {index}: category(source) 검증 실패: {exc}") from exc
