# 설계서 — Common (공통 모듈)

[[Architecture_Design]] 2절(모듈 구조, SOLID)과 3절(Workflow 친화 규약)에서 이름만
정의된 공통 모듈들의 **정확한 필드/시그니처**를 확정하는 설계서. 이 문서를 요구사항
근거로 삼는다 — [[CLAUDE.md]] 3절 순서상 "왜 필요한가"는 [[Architecture_Design]] 2·3절이
이미 답했으므로, 별도 요구사항정의서 없이 이 설계서로 바로 진행한다([[P0_설계서_Logging]]과
동일한 처리 방식).

대상 모듈: `config.py`, `constants.py`, `domain/models.py`, `domain/interfaces.py`,
`exceptions.py`, `workflow/run_context.py`. 모든 Phase(1~5) 코드가 이 모듈들에 의존하므로,
Phase별 코드보다 먼저 구현한다.

## 1. `constants.py` — 도메인 상수

[[CLAUDE.md]] 4절: 환경마다 달라지지 않는, 설계상 고정된 값만 둔다.

```python
CLASS_LABELS = ["IT", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"]  # Scope_Definition 2절
IT_LABEL = "IT"
NON_IT_LABELS = ["DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"]

EMBEDDING_DIM = 1024            # BGE-M3 (AIPro+ 내부 모델) 차원

# 데이터 용도 구분 — Scope_Definition 2.1절 콜렉션명(`<version>_<split>`) 접미사로 사용
DATA_SPLITS = ["train", "test", "validation"]
# 클래스별 분할 비율(3:1:1) — Scope_Definition 3.3절, dataset.split이 참조(P1_설계서_DataPreparation)
SPLIT_RATIOS = {"train": 3, "test": 1, "validation": 1}
# split 이름 → 파일명 stem 매핑("validation"만 "val"로 축약) — cli/run_phase1_5, 향후 embedding/collection이 공유
SPLIT_FILE_STEMS = {"train": "train", "test": "test", "validation": "val"}
# 분할 재현성용 고정 시드 — 값 자체는 임의, "고정되어 있다"는 사실이 핵심
RANDOM_SEED = 42
# 클래스당 레코드 건수 — Scope_Definition 3.3절, dataset.combine 검증 기준(P1_설계서_DataPreparation)
RECORDS_PER_CLASS = 200

# AIPro+ 프로젝트 고정 도메인명 — Scope_Definition 2.1절 "사전 등록 순서" 1단계
DOMAIN_NAME = "embedding_lr"

# JSONL 레코드 키(Scope_Definition 3절 산출물, data.jsonl 실제 키와 일치)
JSON_KEY_QUERY = "질의"
JSON_KEY_RESPONSE = "응답"
JSON_KEY_CATEGORY = "카테고리"
```

- `CLASS_LABELS`의 순서는 `predict_proba()` 출력 순서(사이킷런 `classes_` 정렬,
  알파벳순과 동일)와 일치해야 하므로, `training/trainer.py`는 이 리스트를 직접 하드코딩
  하지 않고 이 상수를 참조한다.
- 5-class와 IT/NON_IT 집계 로직(`evaluation/metrics.py`, `inference/predictor.py`)은
  `IT_LABEL`/`NON_IT_LABELS`만 참조하고 문자열 리터럴을 쓰지 않는다.
- 파일명 `val.jsonl`은 콜렉션명에서는 축약 없이 `validation`으로 표기한다 — `collection.py`가
  파일 stem `val`을 `DATA_SPLITS`의 `validation`으로 매핑한다.

## 2. `config.py` — 환경 설정 (pydantic-settings)

[[CLAUDE.md]] 4절: 환경별로 달라지는 값은 `.env`에서 읽는다.

```python
class Settings(BaseSettings):
    aipro_base_url: str          # 예: http://localhost:28000
    aipro_api_token: str         # Bearer Token — 비밀값, .env 전용, 로그에 절대 출력 금지
    aipro_timeout_seconds: float = 30.0

    embedding_server_base_url: str        # 예: http://localhost:8000 — AIPro+와 무관한 독립 서비스, Phase 5 전용
    embedding_server_timeout_seconds: float = 30.0

    log_level: str = "INFO"      # P0_설계서_Logging 5절
    service_name: str = "embedding_lr"   # P0_설계서_Logging 3절 `service` 필드
    env: str = "local"                    # P0_설계서_Logging 3절 `env` 라벨 후보

    model_dir: str                # model_<ver>.pkl 저장 경로
    status_dir: str = "status"    # Architecture_Design 3절 status/<phase>_<run_id>.json

    model_config = SettingsConfigDict(env_file=".env")
```

- `.env.example`에 키 목록만 값 없이 커밋한다 (`aipro_api_token=` 등).
- `aipro_api_token`은 어떤 로그 필드/예외 메시지에도 절대 포함하지 않는다 —
  [[P0_설계서_Logging]] 3절 `extra` 필드에 실수로 담기지 않도록 로깅 호출부에서 주의.

## 3. `domain/models.py` — 공유 데이터 모델

[[CLAUDE.md]] 2절 등급표 기준 **A등급**(순수 데이터 구조) — 테스트(직렬화/역직렬화,
필드 검증) 먼저 작성.

```python
class QueryRecord(BaseModel):
    """JSONL 레코드 1건 / 추론 요청 1건에 대응"""
    query: str        # JSON_KEY_QUERY
    response: str      # JSON_KEY_RESPONSE
    category: str | None = None   # JSON_KEY_CATEGORY — 학습 데이터는 필수, 추론 요청은 없음(예측 대상)

class KnowledgeRecord(BaseModel):
    """AIPro+ POST /api/rag/knowledge 요청 1건(content 기반) — VectorStore.upsert() 입력
    타입. knowledge_writer.py가 QueryRecord(query/response/category)를 이 형태로 매핑해
    넘긴다. AIPro+가 content로부터 내부적으로 임베딩을 계산해 저장하므로 벡터 필드는
    없다 — 사용자 확인(2026-08-19): "content 받아서 aipro plus 가 vector 생성해서 등록"."""
    content: str
    extended_content: str
    source: str                    # 분류 라벨값 — knowledge_writer가 category를 그대로 매핑

class KnowledgeItem(BaseModel):
    """AIPro+ GET /api/rag/knowledge 응답 1건(임베딩 포함)"""
    id: str
    collection: str
    content: str
    extended_content: str
    domain_id: int
    source: str                    # 분류 라벨값 — KnowledgeRecord.source와 동일 계약
    created_at: str
    embedding: list[float]         # 길이 EMBEDDING_DIM

class PredictionResult(BaseModel):
    """inference/predictor.py 출력, POST /classify 응답 본문"""
    predicted_category: str               # CLASS_LABELS 중 하나
    final_verdict: str                    # "IT" | "NON_IT" — Scope_Definition 2절 최종 판정
    probabilities: dict[str, float]        # {카테고리: 확률}, 5개 키 = CLASS_LABELS
```

- `KnowledgeItem.embedding`의 길이 검증(`EMBEDDING_DIM`)은 pydantic validator로 강제한다
  — 임베딩 API 응답 이상을 조기에 발견하기 위함.
- `PredictionResult.probabilities`의 키 집합은 반드시 `CLASS_LABELS`와 동일해야
  한다(순서는 무관, 키 누락/추가 불가) — validator로 검증.

## 4. `domain/interfaces.py` — Protocol (DIP 경계)

[[CLAUDE.md]] 2절 등급표 기준 **A등급**(Protocol 자체는 로직이 없으므로 테스트 대상은
아니지만, 이를 구현하는 fake는 다른 모듈 테스트에서 사용됨).

```python
class EmbeddingClient(Protocol):
    """독립 Embedding Service(localhost:8000, AIPro+와 무관) 호출 — Phase 5 추론 전용.
    Phase 2는 이 클라이언트를 쓰지 않는다(아래 VectorStore로 대체)."""
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class VectorStore(Protocol):
    """AIPro+(localhost:28000) 지식 데이터 저장소 호출 — Phase 2(학습) 전용."""
    def upsert(self, records: list[KnowledgeRecord], collection: str) -> None:
        """AIPro+ POST /api/rag/knowledge 적재(content 기반 — AIPro+가 내부에서
        임베딩을 계산해 저장). 콜렉션 전체를 재적재하는 방식이므로 레코드 단위
        중복 판별은 하지 않는다."""
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
```

- `EmbeddingClient`와 `VectorStore`는 서로 다른 두 외부 서비스를 감싼다(ISP) —
  `EmbeddingClient`는 독립 Embedding Service의 `POST /embed`를, `VectorStore`는 AIPro+의
  `POST`/`GET /api/rag/knowledge`(upsert + 조회)를 감싼다. `POST /api/rag/knowledge`는
  벡터가 아니라 `content`(텍스트)를 받고 AIPro+가 내부에서 임베딩을 계산해 저장하므로
  (사용자 확인, 2026-08-19), `VectorStore.upsert`의 입력 타입은 벡터가 아니라
  `KnowledgeRecord`(content/extended_content/source)다.
- `EmbeddingClient` 구현체는 `embedding/embedding_server_client.py`(`EmbeddingServerClient`),
  `VectorStore` 구현체는 `embedding/aipro_client.py`(`AIProClient`) — [[Architecture_Design]]
  4절, 둘 다 등급 B(HTTP I/O) — 구현 후 통합 테스트(respx 모킹).
- 이 Protocol들을 테스트에서 fake로 교체해 `embedding/pipeline.py`,
  `training/trainer.py` 등을 실제 AIPro+ 없이 검증한다([[Architecture_Design]] 7절).

## 5. `exceptions.py` — 공통 예외 계층

```python
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
```

- 모든 Phase CLI(`run_phaseN.py`)는 `EmbeddingLRError` 하위 예외만 잡아
  `status/<phase>_<run_id>.json`의 `error` 필드와 [[P0_설계서_Logging]]의 `ERROR` 레벨
  로그에 동일한 메시지를 기록한다. 예상 못한 예외(버그)는 잡지 않고 그대로 전파해
  컨테이너가 비정상 종료하도록 둔다 — 조용한 실패 방지.

## 6. `workflow/run_context.py` — run_id·상태 파일 공통 처리

[[CLAUDE.md]] 2절 등급표 기준 **B등급**(오케스트레이션) — 구현 후 통합 테스트.

```python
def new_run_id() -> str:
    """포맷: <YYYYMMDD-HHMMSS>-<4자리 랜덤 hex>, 예: 20260819-051233-a1b2.
    P0_설계서_Logging 3절 `run_id` 필드와 동일 값을 재사용한다."""

@contextmanager
def run_context(phase: str, settings: Settings):
    """진입 시 status/<phase>_<run_id>.json에 started_at 기록,
    정상 종료 시 ended_at/status=succeeded, 예외 발생 시 status=failed + error 기록.
    logging.LoggerAdapter로 phase/run_id를 모든 로그 라인에 자동 주입."""
```

- 각 `cli/run_phaseN.py`는 `with run_context("phase2", settings) as (run_id, logger): ...`
  형태로 사용 — [[Architecture_Design]] 3절 "모니터링" 규약과 [[P0_설계서_Logging]]의
  `run_id` 필드를 한 곳에서 동시에 구현한다(중복 방지).
- `logging_config.setup_logging()`([[P0_설계서_Logging]] 6절)이 먼저 호출되어 있어야
  `run_context`가 만드는 `LoggerAdapter`가 JSON 포맷터 위에서 동작한다 — 즉
  의존 순서는 `logging_config.setup_logging()` → `run_context()`.

## 7. 구현 순서 및 테스트 등급 요약

| 모듈 | 등급 | 비고 |
|---|---|---|
| `constants.py` | 해당 없음 | 선언만, TDD 대상 아님 |
| `config.py` | B | `.env` 로딩 성공/누락 키 에러 케이스만 통합 테스트 |
| `domain/models.py` | A | 필드 검증/직렬화 — 테스트 먼저, 커버리지 ≥ 90% |
| `domain/interfaces.py` | 해당 없음 | Protocol 선언만 — fake 구현체를 각 소비 모듈 테스트에서 작성 |
| `exceptions.py` | 해당 없음 | 선언만 |
| `workflow/run_context.py` | B | 구현 후 통합 테스트(정상/예외 경로), 커버리지 ≥ 70% |

권장 구현 순서: `constants.py` → `config.py` → `exceptions.py` → `domain/models.py`
(+ 테스트) → `domain/interfaces.py` → `logging_config.py`([[P0_설계서_Logging]]) →
`workflow/run_context.py`(+ 통합 테스트).

## 8. 관련 문서/코드

- 상위 설계: [[Architecture_Design]] 2절(모듈 구조), 3절(Workflow 규약)
- 로깅 표준: [[P0_설계서_Logging]]
- 도메인 요구사항: [[Scope_Definition]] 2절(분류 카테고리), 2.1절(AIPro+ API/콜렉션·적재 전략)
- 관련 코드(계획): `src/embedding_lr/{config,constants,exceptions}.py`,
  `src/embedding_lr/domain/{models,interfaces}.py`, `src/embedding_lr/workflow/run_context.py`
