# P5_요구사항정의서_Inference — Phase 5 요구사항 정의서

[[Scope_Definition]] 5절(추론 파이프라인 설계)과 [[Architecture_Design]] 2절(`inference/
{predictor,api}.py`, `cli/run_inference_server.py` earmark)·6절(`Dockerfile.inference`)을
Phase 5 단위로 구체화한 요구사항 정의서. [[CLAUDE.md]] 3절 "작업 진행 순서"의 1단계
산출물이며, 다음 산출물은 `P5_설계서_Inference.md`(설계서)에서 다룬다.

## 1. 목적 (Why)

Phase 3가 학습하고 Phase 4가 목표치 충족을 확인한 `model_<ver>.pkl`을 실제로 쓰는
**독립 실행형 웹 서비스**를 만든다. 배치 Phase(1~4)와 달리 상시 구동되며, 실시간으로
들어오는 질의를 즉시 분류해 응답한다. 이 서비스가 프로젝트의 최종 산출물([[Scope_Definition]]
6절 "추론 가능한 파이프라인 코드 + 학습된 모델 파일")이다.

## 2. 배경 및 제약 — 임베딩 입력 불일치 발견 및 해소(사용자 확인, 2026-08-20)

- **학습 데이터의 실제 임베딩 대상은 "질의(query) 단독"이다.** Phase 2 구현
  (`embedding/knowledge_writer.py`, [[P2_설계서_Embedding]] 7절)을 보면 AIPro+에
  임베딩 계산을 요청하는 필드는 `content=record.query`(정제된 질의만)이고,
  `extended_content=질의+"\n"+응답`은 저장만 될 뿐 임베딩 계산에 쓰이지 않는다
  (`KnowledgeRecord` 모델 docstring: "AIPro+가 **content**로부터 내부에서 임베딩을
  계산"). 즉 `train/test/val_vectors.parquet`의 임베딩은 전부 질의 단독 텍스트의
  임베딩이다.
- [[Scope_Definition]] 5절 다이어그램은 "입력 텍스트 (질의 + 응답)"을 정제해 그대로
  임베딩하라고 되어 있어 위 사실과 어긋난다 — Phase 2 구현 시점(2026-08-19)에 내려진
  구체적 결정이 Scope_Definition 문서에 소급 반영되지 않은 것으로 보인다.
- **결정(사용자 확인, 2026-08-20)**: Phase 5는 학습된 모델과 입력 분포를 일치시키기
  위해 **질의(query)만 정제해 임베딩**한다. 요청 본문에 `response` 필드가 포함되어도
  분류에는 쓰지 않는다(무시). 이 결정에 따라 이 문서가 [[Scope_Definition]] 5절보다
  **우선**한다 — Scope_Definition 5절의 "질의+응답" 문구는 이 Phase 구현 완료 후 정정
  이슈로 남긴다.
- **독립 웹 서비스, 배치(list) 입출력**: 사용자 확인(2026-08-20)에 따라 Phase 5는 단건이
  아니라 **리스트 형태의 요청을 받아 리스트 형태의 결과를 반환**하는 API로 설계한다.
  요청 리스트의 각 항목과 응답 리스트의 각 항목은 **순서로 대응**한다(응답에 원본 질의를
  에코하지 않음 — `PredictionResult`는 이미 Phase 0에서 확정된 모델이라 이 결정으로
  변경하지 않는다).
- **학습 파이프라인 재사용 극대화**: `preprocessing/text_cleaner.clean_text()`(Phase 2와
  완전히 동일한 정제 규칙)와 `evaluation/metrics.py`의 `probs_to_labels()`(동률 tie-break
  포함)/`to_binary_labels()`(IT/NON_IT 집계)를 그대로 재사용한다 — Phase 4에서 검증에 쓴
  것과 동일한 순수 로직으로 예측 라벨을 뽑으므로, 검증 시점과 추론 시점의 판정 규칙이
  코드 수준에서 100% 동일함이 보장된다([[CLAUDE.md]] 1절 SRP/DRY 취지와 부합, 중복
  구현하지 않음).
- **두 외부 서비스 중 하나만 씀**: Phase 5는 독립 **Embedding Service**(`localhost:8000`,
  `EmbeddingClient` Protocol)만 호출한다. AIPro+(`localhost:28000`)는 전혀 호출하지
  않는다([[Scope_Definition]] 2.1절, [[project_inference_no_qdrant_registration]] 메모와
  일치).
- **모델은 서비스 기동 시 1회만 로드**한다([[Scope_Definition]] 5절) — 요청마다 다시
  읽지 않는다. 로드 실패(`ModelNotFoundError`)는 서비스 기동 자체를 실패시킨다(요청
  단위가 아니라 프로세스 단위 장애로 취급).
- `domain/models.py`의 `PredictionResult`(Phase 0에서 이미 확정: `predicted_category`/
  `final_verdict`/`probabilities`, 상호 일관성 검증 포함)와 `QueryRecord`(`query`/
  `response`/`category: str | None`, 추론 요청은 `category` 미포함)를 그대로 재사용한다
  — 이 Phase에서 이 두 모델에 필드를 추가하거나 변경하지 않는다.

## 3. 범위

### In Scope

| # | 항목 |
|---|---|
| 1 | FastAPI 앱 — `POST /classify`(리스트 요청 → 리스트 응답), `GET /health`(liveness) |
| 2 | `inference/predictor.py` — 정제(`text_cleaner`) → 임베딩(`EmbeddingClient.embed()`, 배치 1회 호출) → 분류(`Classifier.predict_proba()`, 배치 1회 호출) → `evaluation.metrics`로 라벨 판정 → `PredictionResult` 리스트 조립 |
| 3 | 서비스 기동 시 모델 1회 로드(`training.persistence.load_model()`) 및 `EmbeddingClient`/`Classifier` 인스턴스 조립(DIP 경계에서 concrete 구현체 주입) |
| 4 | `cli/run_inference_server.py` — uvicorn 기동 진입점 |
| 5 | `docker/Dockerfile.inference` — fastapi/uvicorn 포함 상시 서비스 이미지 |
| 6 | 요청/응답 스키마(`ClassifyRequest`/`ClassifyResponse`, `domain/models.py` 추가) |

### Out of Scope (다른 Phase 책임 또는 후속 과제)

| 항목 | 담당 |
|---|---|
| 모델 학습·검증 | Phase 3/4 (완료) |
| AIPro+ 등록·조회 | Phase 2 (완료, Phase 5는 호출하지 않음) |
| `docker-compose.yml`(Phase 1~5 통합 오케스트레이션) | 후속 과제 — 이 Phase는 `Dockerfile.inference` 단독 `docker build`/`docker run`으로 검증(Phase 1~4와 동일 관례) |
| Scope_Definition.md 5절 문구 정정 | 후속 문서 작업(2절 참고) |
| 인증/인가, 레이트 리밋 | 미정 — Scope_Definition에 언급 없음, 필요 시 별도 요구사항으로 다룸 |

## 4. 기능 요구사항

### 4.1 API 스키마

**`POST /classify`**

- 요청: `ClassifyRequest { items: list[QueryRecord] }` — 각 `QueryRecord`는 `query`(필수),
  `response`(존재해도 무시, 기존 모델과의 호환을 위해 필드 자체는 유지), `category`(추론
  요청에서는 보통 `None`, 값이 있어도 분류 로직에 영향 없음).
- 응답: `ClassifyResponse { results: list[PredictionResult] }` — `results[i]`는
  `items[i]`에 대응(순서 보장). 빈 리스트 요청 시 빈 리스트 응답.
- `items`가 비어 있지 않은 한 항상 `len(results) == len(items)`.

**`GET /health`**

- 서비스 생존 확인용. 모델이 정상 로드된 상태에서만 200 OK를 반환(로드 자체가
  실패하면 애초에 프로세스가 기동하지 않으므로, 이 엔드포인트가 응답한다는 것 자체가
  모델 로드 성공을 의미한다).

### 4.2 예측 파이프라인 (`inference/predictor.py`)

1. 각 항목의 `query`를 `text_cleaner.clean_text()`로 정제(Phase 2와 동일 규칙 재사용,
   `response`는 사용하지 않음).
2. 정제된 질의 리스트 전체를 `EmbeddingClient.embed()`에 **한 번만** 호출해 1024D 벡터
   리스트를 얻는다(항목별로 나눠 호출하지 않음 — 배치 효율).
3. 벡터 리스트 전체를 `Classifier.predict_proba()`에 **한 번만** 호출해 확률 dict 리스트를
   얻는다.
4. `evaluation.metrics.probs_to_labels()`로 각 항목의 `predicted_category`를 결정(동률
   시 `CLASS_LABELS` 순서 tie-break, Phase 4와 동일 규칙).
5. `evaluation.metrics.to_binary_labels()`로 `final_verdict`(IT/NON_IT)를 결정.
6. 항목별로 `PredictionResult(predicted_category=..., final_verdict=..., probabilities=...)`
   조립.

### 4.3 서비스 기동 (`cli/run_inference_server.py`)

- `Settings()` 로드 → 모델 경로에서 `training.persistence.load_model()`로 모델 1회 로드
  (실패 시 `ModelNotFoundError`로 즉시 기동 실패) → `EmbeddingServerClient(settings)` 생성
  → `inference/predictor.py`의 예측기 조립 → FastAPI 앱에 의존성 주입 → `uvicorn.run()`.
- 로드할 모델 파일 경로, 서비스 host/port는 환경마다 달라지는 값이므로 `.env`(신규
  설정 키)로 관리한다([[CLAUDE.md]] 4절) — 구체적 키 이름/기본값은 설계서에서 확정.

## 5. 비기능 요구사항 (품질 기준)

| 항목 | 기준 |
|---|---|
| 등급/커버리지 | [[CLAUDE.md]] 2절 — `inference/predictor.py`는 등급 B(오케스트레이션, `EmbeddingClient`/`Classifier` Protocol을 fake로 교체해 통합 테스트, ≥70%), `inference/api.py`는 등급 B(FastAPI `TestClient`, ≥70%), `domain/models.py` 추가분(`ClassifyRequest`/`ClassifyResponse`)은 등급 A(≥90%) |
| DIP | `predictor.py`는 `EmbeddingClient`/`Classifier` Protocol에만 의존 — httpx/sklearn을 직접 import하지 않음. `api.py`는 `predictor.py`가 조립한 예측 함수/객체에만 의존 |
| 하드코딩 금지 | 모델 파일 경로, 서비스 host/port를 코드 리터럴로 두지 않음(`.env`, [[CLAUDE.md]] 4절) |
| Docker 실행 | 서비스·테스트 모두 `docker/Dockerfile.inference` 컨테이너 내부에서 실행 가능해야 함 |
| 배치 효율 | 요청 1건(항목 N개)당 `embed()` 1회 + `predict_proba()` 1회로 완료 — 항목 수만큼 반복 호출하지 않음 |
| 응답 일관성 | 동일 요청(동일 정제 결과)에 대해 항상 동일 `PredictionResult`(모델·정제 규칙이 결정적이므로 재현 가능) |

## 6. 산출물

| 파일 | 설명 | 비고 |
|---|---|---|
| `src/embedding_lr/inference/predictor.py` | 예측 파이프라인 오케스트레이션 | 신규 |
| `src/embedding_lr/inference/api.py` | FastAPI 앱(`/classify`, `/health`) | 신규 |
| `src/embedding_lr/cli/run_inference_server.py` | 서비스 기동 진입점 | 신규 |
| `docker/Dockerfile.inference` | 상시 서비스 이미지(fastapi/uvicorn 포함) | 신규 |
| `domain/models.py`(`ClassifyRequest`/`ClassifyResponse` 추가) | API 요청/응답 스키마 | 갱신 |
| `pyproject.toml`(`fastapi`/`uvicorn` 의존성 추가) | 신규 의존성 고정 | 갱신 |

## 7. 완료 기준 (Acceptance Criteria)

- [ ] `POST /classify`에 리스트 요청을 보내면 순서가 대응하는 리스트 응답을 받음(빈
      리스트 포함)
- [ ] 요청의 `response` 필드가 분류 결과에 영향을 주지 않음(동일 `query`, 다른 `response`
      → 동일 결과) 확인
- [ ] `embed()`/`predict_proba()`가 요청 1건당 각각 정확히 1회만 호출됨(fake로 호출
      횟수 검증)
- [ ] `GET /health`가 모델 로드 성공 후 200 OK 반환
- [ ] 모델 로드 실패 시 서비스 기동 자체가 실패함(요청을 받기 전에 실패)
- [ ] `predictor.py`가 `EmbeddingClient`/`Classifier` Protocol에만 의존하고 httpx/sklearn을
      직접 import하지 않음 확인
- [ ] Docker 컨테이너(`Dockerfile.inference`) 내부에서 서비스 기동+테스트 실행 확인
- [ ] 등급 A(`domain/models.py` 추가분) 커버리지 ≥90%, 등급 B(`predictor.py`/`api.py`)
      커버리지 ≥70% 확인(테스트결과서에 기록)

## 8. 리스크 및 참고사항

- 2절에서 발견한 Scope_Definition.md 5절과 실제 구현 간 불일치는 이번 Phase 5
  요구사항으로 해소하지만, Scope_Definition.md 원문 자체는 아직 정정되지 않았다 —
  향후 문서 정리 작업(별도 검토서 또는 Scope_Definition 개정) 필요.
- 만약 이후 "질의+응답을 함께 분류에 반영해야 한다"는 요구가 다시 나오면, Phase 5만
  고쳐서는 안 되고 Phase 2(`knowledge_writer.py`의 `content` 필드)부터 다시 정의해
  train/test/val_vectors.parquet을 재생성하고 Phase 3 모델을 재학습해야 한다(범위가
  Phase 2까지 거슬러 올라감, 2절 참고).
- `docker-compose.yml` 미구성 상태에서는 Embedding Service(`localhost:8000`)가 별도로
  이미 떠 있어야 Phase 5 서비스가 정상 동작한다 — 이 요구사항 문서의 범위는 아니지만
  설계서/테스트결과서에서 전제 조건으로 명시할 것.
