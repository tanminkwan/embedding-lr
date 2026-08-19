"""공유 데이터 모델 — P0_설계서_Common.md 3절. 등급 A(순수 로직) — 테스트 먼저."""

from pydantic import BaseModel, field_validator, model_validator

from embedding_lr.constants import CLASS_LABELS, EMBEDDING_DIM, IT_LABEL


class QueryRecord(BaseModel):
    """JSONL 레코드 1건 / 추론 요청 1건에 대응"""

    query: str
    response: str
    category: str | None = None

    @field_validator("category")
    @classmethod
    def _category_must_be_known_label(cls, value: str | None) -> str | None:
        if value is not None and value not in CLASS_LABELS:
            raise ValueError(f"category must be one of {CLASS_LABELS}, got {value!r}")
        return value


class Domain(BaseModel):
    """AIPro+ 도메인 — POST/GET /api/domains 응답"""

    id: int
    name: str


class Collection(BaseModel):
    """AIPro+ 콜렉션 — POST/GET /api/collections 응답(registration.py가 쓰는 필드만)"""

    name: str
    collection_name: str


class KnowledgeRecord(BaseModel):
    """AIPro+ POST /api/rag/knowledge 요청 1건(content 기반) — knowledge_writer.py가
    QueryRecord를 이 형태로 매핑해 VectorStore.upsert()에 넘긴다. AIPro+가 content로부터
    내부에서 임베딩을 계산해 저장하므로, 이 모델은 벡터를 포함하지 않는다."""

    content: str
    extended_content: str
    source: str

    @field_validator("source")
    @classmethod
    def _source_must_be_known_label(cls, value: str) -> str:
        if value not in CLASS_LABELS:
            raise ValueError(f"source must be one of {CLASS_LABELS}, got {value!r}")
        return value


class KnowledgeItem(BaseModel):
    """AIPro+ GET /api/rag/knowledge 응답 1건(임베딩 포함)"""

    id: str
    collection: str
    content: str
    extended_content: str
    domain_id: int
    source: str
    created_at: str
    embedding: list[float]

    @field_validator("embedding")
    @classmethod
    def _embedding_must_match_embedding_dim(cls, value: list[float]) -> list[float]:
        if len(value) != EMBEDDING_DIM:
            raise ValueError(f"embedding must have length {EMBEDDING_DIM}, got {len(value)}")
        return value


class PredictionResult(BaseModel):
    """inference/predictor.py 출력, POST /classify 응답 본문"""

    predicted_category: str
    final_verdict: str
    probabilities: dict[str, float]

    @field_validator("predicted_category")
    @classmethod
    def _predicted_category_must_be_known_label(cls, value: str) -> str:
        if value not in CLASS_LABELS:
            raise ValueError(f"predicted_category must be one of {CLASS_LABELS}, got {value!r}")
        return value

    @field_validator("final_verdict")
    @classmethod
    def _final_verdict_must_be_it_or_non_it(cls, value: str) -> str:
        if value not in (IT_LABEL, "NON_IT"):
            raise ValueError(f"final_verdict must be one of ['{IT_LABEL}', 'NON_IT'], got {value!r}")
        return value

    @field_validator("probabilities")
    @classmethod
    def _probabilities_keys_must_match_class_labels(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value.keys()) != set(CLASS_LABELS):
            raise ValueError(f"probabilities keys must be exactly {set(CLASS_LABELS)}, got {set(value.keys())}")
        return value

    @model_validator(mode="after")
    def _final_verdict_must_match_predicted_category(self) -> "PredictionResult":
        expected = IT_LABEL if self.predicted_category == IT_LABEL else "NON_IT"
        if self.final_verdict != expected:
            raise ValueError(
                f"final_verdict {self.final_verdict!r} inconsistent with "
                f"predicted_category {self.predicted_category!r} (expected {expected!r})"
            )
        return self
