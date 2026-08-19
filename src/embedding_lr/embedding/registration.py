"""AIPro+ 도메인/콜렉션 사전 등록 보장 — Architecture_Design.md 4절, Scope_Definition.md 2.1절.
등급 B(오케스트레이션) — VectorStore Protocol에만 의존(DIP), 구현 후 fake로 통합 테스트."""

from embedding_lr.domain.interfaces import VectorStore
from embedding_lr.domain.models import Collection, Domain


def ensure_domain(store: VectorStore, name: str) -> Domain:
    """도메인이 이미 존재하면 그대로 반환하고, 없으면 생성한다(idempotent) —
    Scope_Definition.md 2.1절 "사전 등록 순서" ①."""
    for domain in store.list_domains():
        if domain.name == name:
            return domain
    return store.create_domain(name)


def ensure_collection(store: VectorStore, name: str, collection_name: str) -> Collection:
    """콜렉션이 이미 존재하면 그대로 반환하고, 없으면 생성한다(idempotent) —
    Scope_Definition.md 2.1절 "사전 등록 순서" ②. `collection_name`(embedding.collection이
    만든, 점이 치환된 값)으로 존재 여부를 판별한다."""
    for collection in store.list_collections():
        if collection.collection_name == collection_name:
            return collection
    return store.create_collection(name, collection_name)
