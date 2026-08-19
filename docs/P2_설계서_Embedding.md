# 설계서 — Phase 2 임베딩 변환 (Embedding)

[[Architecture_Design]] 2절(모듈 구조)·4절(데이터 흐름 상세)과 [[Scope_Definition]] 2.1절
("AIPro+ API")이 이미 요구사항과 데이터 흐름을 상세히 정의해 두었으므로,
[[P0_설계서_Common]]·[[P2_설계서_TextCleaning]]과 동일한 방식으로 별도 요구사항정의서 없이
이 설계서로 바로 진행한다([[CLAUDE.md]] 3절).

대상 모듈: `embedding/{collection,aipro_client,embedding_server_client,registration,
knowledge_writer,pipeline}.py`, `cli/run_phase2.py`. `preprocessing/text_cleaner.py`는
[[P2_설계서_TextCleaning]]에서 이미 다뤘으므로 이 문서는 그 결과물(정제된 텍스트)을
입력으로 받는 지점부터 다룬다.

## 1. 범위와 설계 전제

- **두 외부 서비스를 명확히 분리한다(ISP)** — [[Scope_Definition]] 2.1절. `VectorStore`
  Protocol(AIPro+, `localhost:28000`)은 **Phase 2(학습)만** 쓰고, `EmbeddingClient`
  Protocol(독립 Embedding Service, `localhost:8000`)은 **Phase 5(추론)만** 쓴다. 이
  문서는 Phase 2 경로(`VectorStore` 쪽)를 중심으로 다루되, 같은 `embedding/` 패키지에
  함께 구현된 `embedding_server_client.py`(`EmbeddingClient` 구현체)도 완결성을 위해
  함께 기록한다 — 실제 호출은 Phase 5(`inference/predictor.py`, 미구현)에서 일어난다.
- **Phase 2는 임베딩을 직접 계산하지 않는다.** `content`(정제된 텍스트)를
  `POST /api/rag/knowledge`로 등록하면 AIPro+가 내부에서 BGE-M3 임베딩을 계산해
  Qdrant에 저장하고, `GET /api/rag/knowledge`로 그 결과를 일괄 조회해서 벡터를 얻는다
  ([[P0_테스트결과서_Common_v2]] 0절 배경 — 사용자 확인, 2026-08-19: "content 받아서
  aipro plus 가 vector 생성해서 등록하는 거야"). 그래서 이 프로젝트 코드 어디에도
  `AIProClient`가 `POST /api/embeddings`(임베딩 단독 계산)를 호출하는 경로는 없다.
- **지식 데이터 등록은 레코드 1건씩 개별 호출한다.** AIPro+는 `POST /api/rag/bulk-upload`도
  제공하지만, 이 프로젝트는 쓰지 않는다(사용자 확인, 2026-08-19) — `AIProClient.upsert()`는
  레코드마다 `POST /api/rag/knowledge`를 반복 호출한다.
- **콜렉션 단위 판별, 레코드 단위 중복 판별 없음** — [[Scope_Definition]] 2.1절. 재실행
  시 콜렉션의 기존 등록 건수와 입력 레코드 수를 비교해 일치하면 재등록을 건너뛰고, 다르면
  콜렉션 전체를 재등록한다. 해시 비교 같은 레코드 단위 중복 판별은 두지 않는다.
- **DIP 경계**: `embedding/pipeline.py`(오케스트레이션)는 `VectorStore`/`DataRepository`
  Protocol에만 의존하고, `AIProClient`/`JsonlRepository` 같은 구체 클래스를 모른다. 테스트는
  이 Protocol들을 fake로 교체해 실제 AIPro+ 없이 수행한다([[Architecture_Design]] 7절).

```
                    ┌─ VectorStore (Protocol, domain/interfaces.py) ────────────┐
                    │  list_domains/create_domain, list_collections/           │
                    │  create_collection, upsert(records, domain_id, coll),    │
                    │  get_knowledge(domain_id, coll, limit)                   │
                    └───────────────────────▲──────────────────────────────────┘
                                             │ 구현
                              AIProClient ───┘ (AIPro+ HTTP, localhost:28000)

embedding.pipeline.run() ── registration.ensure_domain/ensure_collection (idempotent)
                        └── aipro_client.get_knowledge() 건수 비교
                              ├─ 일치 → 그대로 사용
                              └─ 불일치 → text_cleaner.clean_text() → knowledge_writer.write_knowledge()
                                          → aipro_client.get_knowledge() (재조회)
                        └── <split>_vectors.parquet 저장 (embedding + label 컬럼)
```

## 2. `domain/models.py` / `domain/interfaces.py` (갱신)

[[P0_설계서_Common]] 3~4절에 이미 반영됨 — 이 설계서는 Phase 2 관점의 계약만 요약한다.

```python
class KnowledgeRecord(BaseModel):
    """POST /api/rag/knowledge 요청 1건(content 기반). 벡터 필드 없음."""
    content: str
    extended_content: str
    source: str   # CLASS_LABELS 중 하나 — 미확인 값이면 즉시 ValidationError

class KnowledgeItem(BaseModel):
    """GET /api/rag/knowledge 응답 1건(임베딩 포함)."""
    id: str; collection: str; content: str; extended_content: str
    domain_id: int; source: str; created_at: str
    embedding: list[float]   # 길이 EMBEDDING_DIM 아니면 ValidationError

class VectorStore(Protocol):
    def list_domains(self) -> list[Domain]: ...
    def create_domain(self, name: str) -> Domain: ...
    def list_collections(self) -> list[Collection]: ...
    def create_collection(self, name: str, collection_name: str) -> Collection: ...
    def upsert(self, records: list[KnowledgeRecord], domain_id: int, collection: str) -> None: ...
    def get_knowledge(self, domain_id: int, collection: str, limit: int) -> list[KnowledgeItem]: ...
```

- `upsert()`는 최초 [[P0_설계서_Common]] 초안에 `domain_id` 파라미터가 빠져 있었다 —
  AIPro+가 지식 데이터 등록 시 `domain_id`+`collection_name`을 함께 요구한다는 사실이
  확인된 뒤([[Architecture_Design]] 4절) 시그니처를 `upsert(records, domain_id,
  collection)`로 정정했다.

## 3. `embedding/collection.py` — 콜렉션명 생성 (등급 A, 순수 로직)

```python
def collection_name(version: str, split: str) -> str:
    """`<version>_<split>`. version의 `.`을 `_`로 치환(AIPro+ collection_name 패턴
    `^[a-zA-Z0-9_-]+$`, 점 금지, 실제 422 확인됨). split이 DATA_SPLITS에 없으면
    DataValidationError."""

def extract_version_and_split(path: str) -> tuple[str, str]:
    """`data/<version>/{train,test,val}.jsonl` 경로에서 (version, split) 추출.
    파일명이 SPLIT_FILE_STEMS(train/test/val)에 없으면 DataValidationError."""
```

- 외부 의존성 없는 순수 함수 — 예) `("v0.2", "train")` → `"v0_2_train"`.
- `embedding/pipeline.py`가 입력 경로에서 이 두 함수로 콜렉션명을 자동 결정한다 —
  버전·용도를 CLI 인자로 별도로 받지 않는다([[CLAUDE.md]] 4절 하드코딩 금지와도 연결:
  버전 문자열을 경로에서만 얻고 코드에 고정하지 않음).

## 4. `embedding/aipro_client.py` — `VectorStore` 구현체 (등급 B)

```python
class AIProClient:
    def __init__(self, settings: Settings) -> None: ...
    def list_domains(self) -> list[Domain]: ...          # GET /api/domains
    def create_domain(self, name: str) -> Domain: ...     # POST /api/domains
    def list_collections(self) -> list[Collection]: ...   # GET /api/collections
    def create_collection(self, name, collection_name) -> Collection: ...  # POST /api/collections
    def upsert(self, records, domain_id, collection) -> None: ...  # POST /api/rag/knowledge × N
    def get_knowledge(self, domain_id, collection, limit) -> list[KnowledgeItem]: ...  # GET /api/rag/knowledge
    def close(self) -> None: ...
```

- `upsert()`는 `records` 각각에 대해 `POST /api/rag/knowledge`를 개별 호출한다(3절 참고,
  bulk-upload 미사용). 페이로드: `content`/`extended_content`/`source`/`domain_id`/
  `collection_name`.
- 모든 HTTP/파싱 실패는 `AIProClientError`로 통일해서 던진다 — 호출부가 AIPro+ 내부
  구현(httpx 예외 타입 등)을 알 필요가 없게 한다.
- 인증: 이 프로젝트가 접속하는 개발용 AIPro+ 인스턴스는 `Authorization` 헤더 없이도
  동작함이 확인됐지만, 코드는 `Settings.aipro_api_token`으로 Bearer 토큰을 항상
  실어 보낸다 — 운영 환경에서 인증이 요구될 가능성에 대비한 방어적 기본값.

## 5. `embedding/embedding_server_client.py` — `EmbeddingClient` 구현체 (등급 B, Phase 5 전용)

```python
class EmbeddingServerClient:
    def __init__(self, settings: Settings) -> None: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...  # POST /embed, 인증 불필요
    def close(self) -> None: ...
```

- AIPro+와 완전히 무관한 독립 서비스(`Settings.embedding_server_base_url`, 기본
  `localhost:8000`) 호출. Phase 2 파이프라인은 이 클래스를 쓰지 않는다 — Phase 5
  `inference/predictor.py`(미구현)가 쓸 예정이다.
- 실패는 `EmbeddingServerError`로 통일 — `AIProClientError`와 별도 예외 타입으로 분리해,
  로그/에러 핸들링에서 어느 서비스가 실패했는지 구분할 수 있게 한다.

## 6. `embedding/registration.py` — 도메인/콜렉션 사전 등록 보장 (등급 B)

```python
def ensure_domain(store: VectorStore, name: str) -> Domain:
    """존재하면 그대로 반환, 없으면 생성(idempotent)."""

def ensure_collection(store: VectorStore, name: str, collection_name: str) -> Collection:
    """존재하면(collection_name 기준) 그대로 반환, 없으면 생성(idempotent)."""
```

- `VectorStore` Protocol에만 의존 — 실제 HTTP 호출 없이 fake store로 테스트한다.
- `ensure_collection`은 `collection_name` 필드로 존재 여부를 판별한다(`name`은 표시용
  별칭이라 중복 가능 — [[Architecture_Design]] 4절).
- 사전 등록 순서(도메인 → 콜렉션)는 호출부인 `pipeline.py`가 지킨다 — 이 모듈 자체는
  순서를 강제하지 않는다(각 함수가 독립적으로 idempotent이기만 하면 되므로).

## 7. `embedding/knowledge_writer.py` — QueryRecord → KnowledgeRecord 매핑 (등급 B)

```python
def write_knowledge(store: VectorStore, records: list[QueryRecord], domain_id: int, collection: str) -> None:
    """records(이미 정제된 query/response, category 필수)를 KnowledgeRecord로 매핑해
    store.upsert()를 한 번 호출한다. content=query, extended_content=query+"\n"+response,
    source=category. category가 없거나 미확인 라벨이면 DataValidationError(레코드
    인덱스 포함)."""
```

- **텍스트 정제와 결합은 이 모듈의 책임이 아니다** — [[P2_설계서_TextCleaning]] 2절 SRP
  원칙에 따라 `clean_text()` 호출과 query+response 결합은 호출부(`pipeline.py`)가 한다.
  이 모듈은 "이미 정제된 QueryRecord를 AIPro+가 원하는 필드명으로 매핑"만 책임진다.
- `KnowledgeRecord`의 pydantic validator(`source`가 `CLASS_LABELS`에 없으면 거부)가
  던지는 `PydanticValidationError`를 그대로 노출하지 않고 `DataValidationError`로
  감싸서 "레코드 몇 번째"인지 메시지에 포함한다 — `jsonl_repository.py`가 이미 쓰는
  패턴과 동일([[P1_설계서_DataPreparation]] 3절).

## 8. `embedding/pipeline.py` — 전체 오케스트레이션 (등급 B)

```python
def run(repo: DataRepository, store: VectorStore, input_path: str, output_path: str) -> None:
    """input_path(train/test/val.jsonl)를 읽어 AIPro+에 등록·조회한 뒤
    output_path(<split>_vectors.parquet)에 embedding+label을 저장한다."""
```

절차([[Architecture_Design]] 4절 mermaid와 동일):

1. `output_path`가 이미 존재하면 `DataValidationError`([[CLAUDE.md]] 5절 입출력 보존).
2. `repo.load(input_path)` → `list[QueryRecord]`.
3. `collection.extract_version_and_split(input_path)` → `(version, split)` →
   `collection.collection_name(version, split)` → `name`.
4. `registration.ensure_domain(store, DOMAIN_NAME)` → `domain`.
5. `registration.ensure_collection(store, name, name)`.
6. `store.get_knowledge(domain.id, name, limit=len(records)+1)` → `items`.
   - `limit`을 입력 레코드 수보다 1 크게 잡는 이유: 콜렉션에 입력보다 더 많은 건수가
     남아있는 이상 상태까지도 "건수 일치"로 잘못 판단하지 않기 위함.
7. `len(items) == len(records)`면 그대로 8절로 진행(재등록 스킵). 다르면(0건 포함):
   a. 각 레코드의 `query`/`response`를 `clean_text()`로 정제한 새 `QueryRecord` 생성.
   b. `knowledge_writer.write_knowledge(store, cleaned, domain.id, name)`.
   c. `store.get_knowledge(domain.id, name, limit)`으로 재조회.
   d. 재조회 후에도 `len(items) < len(records)`면 `DataValidationError`(재등록 후에도
      불일치 — 정상 흐름에서는 발생하지 않아야 하는 방어적 체크).
8. `items`의 `embedding`/`source`를 각각 `embedding`/`label` 컬럼으로 하는
   `pandas.DataFrame`을 만들어 `output_path`에 parquet으로 저장(pyarrow 엔진).
9. 이 함수는 **어느 분기에서도 `embed()`를 호출하지 않는다** — 벡터는 항상 AIPro+가
   계산해 저장한 것을 `get_knowledge()`로 가져온다.

## 9. `cli/run_phase2.py`

```
Trigger: python -m embedding_lr.cli.run_phase2 --input <path> --output <path>
Input:   --input(train/test/val.jsonl 경로 1개)
Output:  --output(<split>_vectors.parquet 경로 1개) — 이미 존재하면 실패
```

- `run_phase1(_5).py`와 동일한 컨벤션: `Settings()` → `setup_logging()` →
  `run_context("phase2", settings)`로 상태 파일(`status/phase2_<run_id>.json`) 기록.
- `JsonlRepository()` + `AIProClient(settings)`를 `pipeline.run()`에 주입하고,
  `finally`에서 `AIProClient.close()`로 HTTP 커넥션을 정리한다.
- **split별 독립 실행**이 원칙이므로([[Architecture_Design]] 4절 "split별 독립 실행"),
  한 번의 CLI 실행은 `train.jsonl`/`test.jsonl`/`val.jsonl` 중 하나만 처리한다 —
  세 파일을 모두 처리하려면 CLI를 3번 실행한다.

## 10. 데이터 흐름 요약

```
data/<version>/{train,test,val}.jsonl
   ──(collection.extract_version_and_split + collection_name)──> <version>_<split>
   ──(registration.ensure_domain/ensure_collection)──> AIPro+ 도메인/콜렉션 보장
   ──(aipro_client.get_knowledge, 건수 비교)──┐
        일치 → 재등록 스킵 ──────────────────┤
        불일치 → text_cleaner.clean_text()   │
              → knowledge_writer.write_knowledge (AIPro+ POST /api/rag/knowledge × N)
              → aipro_client.get_knowledge (재조회) ┘
   ──(pandas → parquet)──> <version>/<split>_vectors.parquet (embedding + label)
                                                          [cli/run_phase2.py]
```

## 11. 테스트 등급 및 완료 기준

| 모듈 | 등급([[CLAUDE.md]] 2절) | 목표 커버리지 | 비고 |
|---|---|---|---|
| `embedding/collection.py` | A | ≥ 90% | 순수 로직 — 테스트 먼저, 점 치환/미확인 split 케이스 포함 |
| `embedding/aipro_client.py` | B | ≥ 70% | HTTP I/O — respx 모킹, happy path + HTTP 에러 + 응답 파싱 실패 |
| `embedding/embedding_server_client.py` | B | ≥ 70% | HTTP I/O — respx 모킹, happy path + 에러 케이스 (Phase 5 전용, 이 Phase에서는 직접 쓰이지 않음) |
| `embedding/registration.py` | B | ≥ 70% | fake `VectorStore` — idempotent 판단(존재/미존재) |
| `embedding/knowledge_writer.py` | B | ≥ 70% | fake `VectorStore` — 매핑 정확성 + category 누락 에러 |
| `embedding/pipeline.py` | B | ≥ 70% | fake `DataRepository`/`VectorStore` — 스킵/재등록/출력 존재/재등록 후 불일치 4가지 분기 |
| `cli/run_phase2.py` | B | ≥ 70% | respx로 AIPro+ 전체 흐름 모킹 — 실제 네트워크 호출 없음 |

완료 기준: 위 6개 모듈 + CLI가 각 목표 커버리지를 충족하고, `embed()` 미호출·
bulk-upload 미사용·재등록 스킵 로직이 테스트로 확인되면 Phase 2 코드는 완료로
간주한다. 실측치는 [[P2_테스트결과서_Embedding]] 참고.

## 12. 관련 문서/코드

- 요구사항/데이터 흐름: [[Scope_Definition]] 2.1절, [[Architecture_Design]] 2절/4절
- 공통 모듈: [[P0_설계서_Common]] 3절(`KnowledgeRecord`/`KnowledgeItem`), 4절(`VectorStore`/
  `EmbeddingClient` Protocol), [[P0_테스트결과서_Common_v2]](모델 변경 배경)
- 텍스트 정제: [[P2_설계서_TextCleaning]]
- 관련 코드: `src/embedding_lr/embedding/{collection,aipro_client,embedding_server_client,
  registration,knowledge_writer,pipeline}.py`, `src/embedding_lr/cli/run_phase2.py`
- 테스트: `tests/unit/test_collection.py`, `tests/integration/test_{aipro_client,
  embedding_server_client,registration,knowledge_writer,pipeline,run_phase2}.py`
