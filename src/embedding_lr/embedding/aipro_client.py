"""domain.interfaces.VectorStore 구현체 — AIPro+(localhost:28000) HTTP 클라이언트.
P0_설계서_Common.md 4절. 등급 B(오케스트레이션, HTTP I/O) — 구현 후 통합 테스트(respx 모킹)."""

import httpx
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from embedding_lr.config import Settings
from embedding_lr.domain.models import Collection, Domain, KnowledgeItem, KnowledgeRecord
from embedding_lr.exceptions import AIProClientError


class AIProClient:
    """AIPro+(localhost:28000) API 호출을 감싼다 — Scope_Definition 2.1절."""

    def __init__(self, settings: Settings) -> None:
        headers = {}
        if settings.aipro_api_token:
            headers["Authorization"] = f"Bearer {settings.aipro_api_token}"
        self._client = httpx.Client(
            base_url=settings.aipro_base_url,
            headers=headers,
            timeout=settings.aipro_timeout_seconds,
        )

    def list_domains(self) -> list[Domain]:
        """GET /api/domains — 등록된 도메인 전체 조회."""
        return self._get_list("/api/domains", Domain)

    def create_domain(self, name: str) -> Domain:
        """POST /api/domains — 도메인 생성."""
        return self._post("/api/domains", {"name": name}, Domain)

    def list_collections(self) -> list[Collection]:
        """GET /api/collections — 등록된 콜렉션 전체 조회."""
        return self._get_list("/api/collections", Collection)

    def create_collection(self, name: str, collection_name: str) -> Collection:
        """POST /api/collections — 콜렉션 생성. `collection_name`은 AIPro+가
        `^[a-zA-Z0-9_-]+$`만 허용한다(점 금지, 실제 422 확인됨) — 호출부(embedding.collection)가
        사전에 치환해서 넘겨야 한다."""
        return self._post(
            "/api/collections", {"name": name, "collection_name": collection_name}, Collection
        )

    def upsert(self, records: list[KnowledgeRecord], domain_id: int, collection: str) -> None:
        """POST /api/rag/knowledge — 레코드 1건씩 개별 등록. `/api/rag/bulk-upload`(벌크
        업로드)는 쓰지 않는다(사용자 확인, 2026-08-19) — AIPro+가 각 요청의 content로부터
        내부에서 임베딩을 계산해 저장한다."""
        for record in records:
            payload = {
                "content": record.content,
                "extended_content": record.extended_content,
                "source": record.source,
                "domain_id": domain_id,
                "collection_name": collection,
            }
            try:
                response = self._client.post("/api/rag/knowledge", json=payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise AIProClientError(f"POST /api/rag/knowledge 실패: {exc}") from exc

    def get_knowledge(self, domain_id: int, collection: str, limit: int) -> list[KnowledgeItem]:
        """GET /api/rag/knowledge — domain_id+collection 조건, 임베딩 포함 조회.
        embedding/pipeline.py가 지식 데이터를 일괄 등록(POST, content 기반)한 뒤,
        AIPro+가 내부에서 계산해 저장한 임베딩을 이 API로 일괄 조회해 parquet을 만든다
        — Architecture_Design.md 4절 "Phase 2 임베딩 파이프라인"."""
        return self._get_list(
            "/api/rag/knowledge",
            KnowledgeItem,
            params={"domain_id": domain_id, "collection": collection, "limit": limit},
        )

    def _get_list(self, path: str, model: type[BaseModel], params: dict | None = None) -> list:
        try:
            response = self._client.get(path, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AIProClientError(f"GET {path} 실패: {exc}") from exc

        try:
            return [model(**item) for item in response.json()]
        except (PydanticValidationError, TypeError, ValueError) as exc:
            raise AIProClientError(f"GET {path} 응답 파싱 실패: {exc}") from exc

    def _post(self, path: str, payload: dict, model: type[BaseModel]):
        try:
            response = self._client.post(path, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AIProClientError(f"POST {path} 실패: {exc}") from exc

        try:
            return model(**response.json())
        except (PydanticValidationError, TypeError, ValueError) as exc:
            raise AIProClientError(f"POST {path} 응답 파싱 실패: {exc}") from exc

    def close(self) -> None:
        self._client.close()
