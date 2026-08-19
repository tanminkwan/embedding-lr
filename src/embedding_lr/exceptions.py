"""공통 예외 계층 — P0_설계서_Common.md 5절."""


class EmbeddingLRError(Exception):
    """프로젝트 공통 베이스 예외"""


class AIProClientError(EmbeddingLRError):
    """AIPro+ API 호출 실패(HTTP 오류, 타임아웃, 응답 스키마 불일치)"""


class EmbeddingServerError(EmbeddingLRError):
    """독립 Embedding Service(AIPro+와 별개, Phase 5 추론 전용) 호출 실패
    (HTTP 오류, 타임아웃, 응답 스키마 불일치)"""


class ModelNotFoundError(EmbeddingLRError):
    """model_<ver>.pkl 로드 실패 — 추론 서비스 기동 시"""


class DataValidationError(EmbeddingLRError):
    """JSONL 스키마/라벨 값 불일치 — dataset.combine/split 단계"""
