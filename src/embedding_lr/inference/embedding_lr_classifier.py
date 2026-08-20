"""TextClassifier Protocol 구현체 — 임베딩+LR 조합 — P5_설계서_Inference.md 3절. 등급 B
(오케스트레이션) — EmbeddingClient/Classifier Protocol에만 의존, fake로 통합 테스트.

이 모듈이 EmbeddingClient와 Classifier 두 Protocol을 동시에 아는 유일한 곳이다 — 두
하위 Protocol을 조합해 상위 TextClassifier Protocol 하나로 노출하는 어댑터이기 때문이다.
"""

from embedding_lr.domain.interfaces import Classifier, EmbeddingClient


class EmbeddingLRTextClassifier:
    """TextClassifier Protocol 구현체 — EmbeddingClient.embed() + Classifier.predict_proba()
    조합(현재 유일한 프로덕션 백엔드)."""

    def __init__(self, embedding_client: EmbeddingClient, model: Classifier) -> None:
        self._embedding_client = embedding_client
        self._model = model

    def classify(self, queries: list[str]) -> list[dict[str, float]]:
        """queries가 비어 있으면 embed()/predict_proba() 호출 없이 빈 리스트 반환."""
        if not queries:
            return []
        vectors = self._embedding_client.embed(queries)
        return self._model.predict_proba(vectors)
