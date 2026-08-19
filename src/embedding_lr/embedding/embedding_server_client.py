"""domain.interfaces.EmbeddingClient 구현체 — 독립 Embedding Service HTTP 클라이언트.
AIPro+(localhost:28000)와는 별개의 서비스(예: localhost:8000)이며, Phase 5 추론 전용이다
— Phase 2는 AIPro+ 일괄 등록/조회(embedding/aipro_client.py)로 벡터를 얻는다.
P0_설계서_Common.md 4절. 등급 B(오케스트레이션, HTTP I/O) — 구현 후 통합 테스트(respx 모킹)."""

import httpx

from embedding_lr.config import Settings
from embedding_lr.exceptions import EmbeddingServerError


class EmbeddingServerClient:
    """독립 Embedding Service API 호출을 감싼다 — 인증 불필요, 단건/소량 실시간 임베딩용."""

    def __init__(self, settings: Settings) -> None:
        self._client = httpx.Client(
            base_url=settings.embedding_server_base_url,
            timeout=settings.embedding_server_timeout_seconds,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """POST /embed — 텍스트 목록 → 임베딩 벡터 배열. 추론 경로는 Qdrant 저장이
        필요 없으므로 AIPro+를 거치지 않고 이 서비스를 직접 호출한다."""
        try:
            response = self._client.post("/embed", json={"texts": texts})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingServerError(f"POST /embed 실패: {exc}") from exc

        try:
            return response.json()["embeddings"]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingServerError(f"POST /embed 응답 파싱 실패: {exc}") from exc

    def close(self) -> None:
        self._client.close()
