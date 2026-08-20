# 설계서 — Phase 5 추론 (Inference)

[[P5_요구사항정의서_Inference]](요구사항정의서)를 [[Architecture_Design]] 2절(모듈 구조,
`inference/{predictor,api}.py`/`cli/run_inference_server.py` earmark)·6절(`Dockerfile.
inference`)의 실제 모듈 시그니처로 구체화한 설계서. [[CLAUDE.md]] 3절 순서상 2단계
산출물이다.

대상 모듈(신규): `inference/{predictor,api,embedding_lr_classifier}.py`,
`cli/run_inference_server.py`, `docker/Dockerfile.inference`. 대상 모듈(갱신):
`domain/interfaces.py`(`TextClassifier` Protocol), `domain/models.py`(`ClassifyRequest`/
`ClassifyResponse`), `config.py`(`model_path`/`inference_host`/`inference_port`),
`pyproject.toml`(`inference` extras 그룹).

**분류 방식 추상화(사용자 확인, 2026-08-20)**: 지금은 "임베딩+LR" 한 가지 방식뿐이지만,
추후 NLI 등 다른 방식이 추가되거나 여러 방식을 조합(앙상블)할 수 있다는 요구에 따라,
`inference/predictor.py`/`api.py`가 "어떻게 분류하는지"를 전혀 모르도록 한 단계 더
추상화한다(1절 참고). 이 설계는 요구사항 4.2절이 원래 기술한 "정제→임베딩→분류"
파이프라인을 **삭제하지 않고**, 그 전체를 `TextClassifier`라는 하나의 Protocol 뒤로
캡슐화하는 것이다.

## 1. 범위와 설계 전제

### 1.1 분류 방식 추상화 — `TextClassifier` Protocol

기존 계획(`inference/predictor.py`가 `EmbeddingClient`+`Classifier` 두 Protocol에 직접
의존)은 "텍스트를 분류한다"는 행위 안에 "임베딩을 거친다"는 구현 세부사항이 새어
들어가 있었다(ISP 위반 소지). 이를 한 단계 위 인터페이스로 분리한다.

```python
# domain/interfaces.py (갱신)
class TextClassifier(Protocol):
    """(정제된) 텍스트 리스트 → 클래스별 확률 dict 리스트. 임베딩+LR 조합은 이 Protocol의
    구현체 중 하나(EmbeddingLRTextClassifier)일 뿐이다 — NLI 등 임베딩을 거치지 않는
    방식이나, 여러 구현체를 조합한 앙상블도 이 Protocol 하나만 만족하면 predictor.py/
    api.py의 수정 없이 교체·추가할 수 있다."""

    def classify(self, queries: list[str]) -> list[dict[str, float]]: ...
```

- **기존 `Classifier` Protocol(embedding vector → 확률, `training/trainer.py`가 구현)은
  그대로 유지한다** — 없애거나 대체하지 않는다. `Classifier`는 "임베딩 공간에서의
  분류기"라는 더 좁은 개념이고, `TextClassifier`는 "원문 텍스트에서의 분류기"라는 더
  넓은 개념이다. 이 둘의 관계는 2절의 `EmbeddingLRTextClassifier`가 어댑터로 연결한다.
- `inference/predictor.py`는 이제 `TextClassifier` **하나**에만 의존한다(더 이상
  `EmbeddingClient`/`Classifier`를 직접 알지 않는다) — 4절 참고.
- **확장 시나리오(이번 Phase에서 구현하지 않음, 설계만 열어 둠)**:
  - NLI 등 임베딩을 쓰지 않는 방식: `NLITextClassifier`가 `classify()`를 직접 구현(내부에서
    무엇을 호출하든 `predictor.py`/`api.py`는 알 필요 없음).
  - 복수 방식 조합(앙상블): `EnsembleTextClassifier(classifiers: list[TextClassifier])`도
    같은 `TextClassifier`를 구현하므로(재귀적 합성), 여러 `classify()` 결과를 모아 평균/투표
    등으로 합치기만 하면 되고 `predictor.py`/`api.py`는 여전히 무수정이다.
  - 어느 경우든 실제 교체 지점은 `cli/run_inference_server.py`(합성 루트) **한 곳**뿐이다
    (6절).

### 1.2 그 외 설계 전제

- **임베딩 입력은 질의(query) 단독**([[P5_요구사항정의서_Inference]] 2절, 실제 AIPro+
  대상 실험으로 검증 완료) — `EmbeddingLRTextClassifier`는 `item.response`를 읽지 않는다
  (애초에 `TextClassifier.classify()`는 정제된 질의 문자열만 받으므로 `response`가 이
  경계를 넘어올 수 없다 — 구조적으로 차단됨).
- **DIP**: concrete 구현체(`EmbeddingServerClient`, `LogisticRegressionClassifier`,
  `EmbeddingLRTextClassifier`)는 `cli/run_inference_server.py`(합성 루트)에서만 조립한다
  — Phase 1~4의 `cli/run_phaseN.py`가 concrete 구현체를 조립하는 것과 동일한 패턴.
- **모델은 5-class 전부를 학습했다고 가정**한다 — Phase 3 파이프라인은 항상 5-class
  균등 데이터로 학습하므로 `predict_proba()`가 반환하는 dict는 항상 `CLASS_LABELS` 5개
  키를 전부 포함한다. `PredictionResult.probabilities` validator가 이미 "정확히
  `CLASS_LABELS` 집합과 일치"를 강제하므로, 이 가정이 깨지면 응답 생성 시점에
  `ValidationError`로 즉시 드러난다 — 별도 방어 로직을 추가하지 않는다.
- **배치 처리, 순서 대응**: `ClassifyRequest.items`(N개) → `TextClassifier.classify()`
  1회 → `ClassifyResponse.results`(N개), `results[i]`는 `items[i]`에 대응.
- **재사용(신규 로직 최소화)**: `preprocessing.text_cleaner.clean_text()`(Phase 2와 동일),
  `evaluation.metrics.probs_to_labels()`/`to_binary_labels()`(Phase 4와 동일 판정 로직)를
  그대로 가져다 쓴다.
- **모델 로드는 서비스 기동 시 1회**(`training.persistence.load_model()`) — 로드 실패
  (`ModelNotFoundError`)는 `cli/run_inference_server.main()`에서 전파되어 프로세스
  자체가 뜨지 않는다.
- **알려진 예외의 HTTP 상태코드 매핑**: `EmbeddingServerError`(Embedding Service 장애)는
  503으로 매핑한다 — sklearn 예외 등 예기치 못한 오류는 FastAPI 기본 처리(500)에 맡긴다
  (과설계 방지, [[CLAUDE.md]] 4절).

```
                              ┌─ TextClassifier (Protocol) ────────┐
                              │  classify(queries) -> list[probs]  │
                              └──────────────────▲──────────────────┘
                                                  │ 구현(현재 유일한 프로덕션 백엔드)
                                    EmbeddingLRTextClassifier
                                          │ 내부에서 조합
                     ┌────────────────────┴────────────────────┐
              ┌─ EmbeddingClient ─┐                    ┌─ Classifier ─┐
              │ embed(texts)      │                    │ predict_proba │
              └─────────▲─────────┘                    └───────▲───────┘
                        │ 구현(재사용)                            │ 구현(재사용)
             EmbeddingServerClient                LogisticRegressionClassifier

cli.run_inference_server.main()
    Settings() → persistence.load_model(settings.model_path) → model: Classifier
               → EmbeddingServerClient(settings) → embedding_client: EmbeddingClient
               → EmbeddingLRTextClassifier(embedding_client, model) → classifier: TextClassifier
               → inference.api.create_app(classifier) → app: FastAPI
               → uvicorn.run(app, host=settings.inference_host, port=settings.inference_port)

POST /classify
    ClassifyRequest{items} ──> inference.predictor.predict(classifier, items)
        text_cleaner.clean_text(item.query) for each item  ──> cleaned queries
        classifier.classify(cleaned queries)                ──> probs (1회 호출, 내부 구현은 predictor.py가 모름)
        evaluation.metrics.probs_to_labels(probs)            ──> predicted_category 리스트
        evaluation.metrics.to_binary_labels(predicted)       ──> final_verdict 리스트
    ──> list[PredictionResult] ──> ClassifyResponse{results}
```

## 2. `domain/interfaces.py` (갱신) — `TextClassifier` Protocol 추가

1.1절 코드 그대로 추가한다. 기존 `EmbeddingClient`/`VectorStore`/`Classifier`/
`DataRepository`는 변경하지 않는다.

## 3. `inference/embedding_lr_classifier.py` (신규, 등급 B) — 임베딩+LR 어댑터

```python
class EmbeddingLRTextClassifier:
    """TextClassifier Protocol 구현체 — EmbeddingClient.embed() + Classifier.predict_proba()
    조합(현재 유일한 프로덕션 백엔드). 이름 그대로 '임베딩+LR 방식'임을 명시해, 나중에
    NLITextClassifier 등 다른 구현체와 나란히 존재해도 헷갈리지 않게 한다."""

    def __init__(self, embedding_client: EmbeddingClient, model: Classifier) -> None:
        self._embedding_client = embedding_client
        self._model = model

    def classify(self, queries: list[str]) -> list[dict[str, float]]:
        """queries가 비어 있으면 embed()/predict_proba() 호출 없이 빈 리스트 반환."""
        if not queries:
            return []
        vectors = self._embedding_client.embed(queries)
        return self._model.predict_proba(vectors)
```

- 이 파일이 `EmbeddingClient`/`Classifier` 두 Protocol을 **동시에** 아는 유일한 곳이다
  — 어댑터의 존재 이유 자체가 "두 하위 Protocol을 조합해 상위 Protocol 하나로 노출"이므로,
  두 하위 Protocol을 아는 것은 위반이 아니라 이 모듈의 책임이다(SRP).
- 등급 B(오케스트레이션) — `trainer.py`/`predictor.py`와 같은 성격: 결정적이지만 외부
  Protocol 호출을 조합한다.

## 4. `inference/predictor.py` (신규, 등급 B — [[CLAUDE.md]] 2절 표에 이미 명시된 분류)

```python
def predict(classifier: TextClassifier, items: list[QueryRecord]) -> list[PredictionResult]:
    """items가 비어 있으면 빈 리스트를 즉시 반환(classifier.classify() 호출 없음). 아니면:
    1) 각 item.query를 text_cleaner.clean_text()로 정제(item.response는 읽지 않음)
    2) classifier.classify(cleaned_queries) 1회 호출 3) evaluation.metrics.probs_to_labels(
    probs)로 predicted_category 리스트, evaluation.metrics.to_binary_labels(predicted)로
    final_verdict 리스트를 얻어 zip으로 PredictionResult 리스트 조립."""
```

- **`TextClassifier` Protocol 하나에만 의존** — 임베딩을 쓰는지, LR인지, NLI인지, 앙상블
  인지 이 함수는 전혀 모른다. httpx/sklearn 어느 쪽도 직접 import하지 않는다.
- `evaluation.metrics`를 import하는 것은 [[CLAUDE.md]] 1절 SRP 위반이 아니다 —
  `metrics.py` 자체가 이미 순수 함수(등급 A)라 어느 계층에서 가져다 써도 안전하다. Phase
  4(`evaluation/report.py`)와 Phase 5(`inference/predictor.py`)가 같은 판정 함수를
  공유하므로, 검증 시점 결과와 실제 서빙 시점 결과가 코드 수준에서 어긋날 수 없다 —
  분류 방식이 나중에 바뀌어도(1.1절) 이 판정 로직(argmax tie-break, IT/NON_IT 집계)은
  그대로 유지된다.

## 5. `inference/api.py` (신규, 등급 B)

```python
def create_app(classifier: TextClassifier) -> FastAPI:
    """FastAPI 앱을 조립해 반환하는 팩토리 함수 — 전역 상태를 두지 않고 클로저로
    classifier를 캡처한다. 테스트 시 fake TextClassifier를 주입해 TestClient로 검증
    가능. 인자가 TextClassifier 하나뿐이므로, 조립부(6절)가 어떤 백엔드를 넣어주든
    이 함수는 무수정이다."""

    app = FastAPI()

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(request: ClassifyRequest) -> ClassifyResponse:
        results = predictor.predict(classifier, request.items)
        return ClassifyResponse(results=results)

    @app.exception_handler(EmbeddingServerError)
    def handle_embedding_server_error(request: Request, exc: EmbeddingServerError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
```

- `EmbeddingServerError`를 여기서 여전히 매핑하는 이유: 지금 프로덕션 백엔드
  (`EmbeddingLRTextClassifier`)가 내부적으로 `EmbeddingClient`를 쓰기 때문에 이 예외가
  실제로 발생할 수 있다. 이후 임베딩을 쓰지 않는 백엔드(NLI 등)로 완전히 교체된다면 이
  매핑은 자연히 발생하지 않게 되므로, 지금 남겨 둬도 해가 없다(과거 코드 삭제는 그때
  가서 판단).
- `create_app()`을 함수로 둔 이유: 모듈 전역에 `classifier`를 두면 테스트마다 모듈을
  재로드하거나 monkeypatch해야 해서 지저분해진다 — 팩토리 패턴으로 의존성을 인자로
  받으면 `TestClient(create_app(fake_classifier))`처럼 테스트가 단순해진다.

## 6. `config.py` (갱신) — 추론 서비스 설정

```python
class Settings(BaseSettings):
    ...
    model_path: str
    inference_host: str = "0.0.0.0"
    inference_port: int = 8080
```

- `model_path`: 로드할 특정 모델 파일 경로(`model_dir` 하위 특정 `.pkl` 파일) — 환경마다
  달라지므로 필수 필드, 기본값 없음([[CLAUDE.md]] 4절). 기존 `model_dir`(Phase 3가 산출물을
  쓰는 디렉터리)과는 별개.
- `inference_host`/`inference_port`: 컨테이너 내부 바인딩 주소/포트. 기본값은 Docker
  컨테이너 관례(`0.0.0.0`, `8080`)로 두고 `.env`로 재정의 가능.

## 7. `cli/run_inference_server.py` — 합성 루트(composition root)

```
Trigger: python -m embedding_lr.cli.run_inference_server
Input:   환경변수(.env) — MODEL_PATH(필수), INFERENCE_HOST/INFERENCE_PORT(선택),
         EMBEDDING_SERVER_BASE_URL(기존 설정 재사용)
Output:  없음(상시 프로세스) — 기동 실패 시 프로세스가 즉시 종료되고 스택트레이스 출력
```

```python
def main() -> None:
    settings = Settings()
    setup_logging(settings)

    model = persistence.load_model(settings.model_path)
    embedding_client = EmbeddingServerClient(settings)
    classifier = EmbeddingLRTextClassifier(embedding_client, model)
    app = create_app(classifier)

    uvicorn.run(app, host=settings.inference_host, port=settings.inference_port)
```

- **이 파일이 "임베딩+LR 방식을 쓴다"는 결정이 코드에 등장하는 유일한 곳이다.** 나중에
  NLI로 바꾸거나 앙상블을 붙이려면 이 함수의 3줄(`model`/`embedding_client`/`classifier`
  조립 부분)만 교체하면 되고, `predictor.py`/`api.py`/`domain/interfaces.py`(Protocol
  정의 제외)는 그대로 둔다.
- `run_phase1~4.py`와 달리 `run_context()`(status 파일 기록)를 쓰지 않는다 — 그건 "시작
  하고 끝나는 배치 작업"을 위한 것이고, 이 프로세스는 종료 시점이 없는 상시 서비스라
  started_at/ended_at 개념이 맞지 않는다([[Architecture_Design]] 1절과 일치). 대신
  uvicorn 자체 접근 로그 + `logging_config.py` 표준 로거로 기동/요청 로그를 남긴다.

## 8. `docker/Dockerfile.inference` (신규) + `pyproject.toml` (갱신)

```dockerfile
# Phase 5 상시 서비스 이미지 — Architecture_Design.md 6절
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[dev,inference]"

COPY tests ./tests

EXPOSE 8080
CMD ["python", "-m", "embedding_lr.cli.run_inference_server"]
```

```toml
[project.optional-dependencies]
dev = [...]                    # 기존 그대로
inference = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
]
```

- `config/` 디렉터리는 COPY하지 않는다 — Phase 5는 JSON 설정 파일을 읽지 않고 `.env`만
  쓴다([[P5_요구사항정의서_Inference]] 3절).
- `fastapi`/`uvicorn`을 `dependencies`가 아니라 별도 `inference` extras로 둔 이유: Phase
  1~4 배치 이미지(`Dockerfile.pipeline`)가 이 두 패키지를 설치할 이유가 없다.
- `docker-compose.yml`은 이번 Phase에서 만들지 않는다([[P5_요구사항정의서_Inference]]
  3절 Out of Scope) — `docker build -f docker/Dockerfile.inference`와 `docker run`으로
  단독 검증한다(Phase 1~4와 동일 관례).

## 9. 데이터 흐름 요약

```
.env(MODEL_PATH, INFERENCE_HOST/PORT, EMBEDDING_SERVER_BASE_URL)
        │
cli.run_inference_server.main()
        │  persistence.load_model(MODEL_PATH) ──> model: Classifier (1회, 기동 시)
        │  EmbeddingServerClient(settings)     ──> embedding_client: EmbeddingClient (1회)
        │  EmbeddingLRTextClassifier(embedding_client, model) ──> classifier: TextClassifier
        ▼
inference.api.create_app(classifier) ──> FastAPI app ──> uvicorn.run(...)
        │
        │  (요청마다)
        ▼
POST /classify  ClassifyRequest{items} ──> predictor.predict(classifier, items)
        │   text_cleaner.clean_text(item.query) ×N
        │   classifier.classify([...]) ──(EmbeddingLRTextClassifier 내부)──>
        │       embedding_client.embed([...]) ──> Embedding Service POST /embed (1회)
        │       model.predict_proba([...]) (1회)
        │   evaluation.metrics.probs_to_labels/to_binary_labels
        ▼
        list[PredictionResult] ──> ClassifyResponse{results} ──> HTTP 200 응답
```

## 10. 테스트 등급 및 완료 기준

| 모듈 | 등급([[CLAUDE.md]] 2절) | 목표 커버리지 | 비고 |
|---|---|---|---|
| `domain/models.py`(`ClassifyRequest`/`ClassifyResponse` 추가분) | A | ≥ 90% | 순수 래퍼 모델 — 빈 리스트/정상 리스트 생성 |
| `inference/embedding_lr_classifier.py` | B | ≥ 70% | fake `EmbeddingClient`/`Classifier`로 통합 테스트 — 빈 리스트 시 `embed()`/`predict_proba()` 미호출, 정상 입력 시 각각 정확히 1회 호출(call count 검증) |
| `inference/predictor.py` | B | ≥ 70% | fake `TextClassifier`로 통합 테스트 — 빈 리스트 시 `classify()` 미호출, `response` 필드가 결과에 영향 없음, `classify()`가 정확히 1회만 호출됨, 순서 대응 확인 |
| `inference/api.py` | B | ≥ 70% | FastAPI `TestClient` + fake `TextClassifier` — `/classify` 정상 응답, 빈 리스트 요청, `EmbeddingServerError` 발생 시 503, `/health` 200 |
| `cli/run_inference_server.py` | B | ≥ 70% | fixture 모델 `.pkl` + monkeypatch로 `uvicorn.run` 호출 인자(host/port/app) 검증, 모델 파일 부재 시 `ModelNotFoundError`로 기동 실패 확인 |

완료 기준: 위 5개 대상이 각 목표 커버리지를 충족하고, (1) `response` 필드가 분류 결과에
영향을 주지 않음, (2) `embed()`/`predict_proba()`가 요청 1건당 각 1회만 호출됨, (3)
`predictor.py`/`api.py`가 `TextClassifier` Protocol에만 의존하고 httpx/sklearn을 직접
import하지 않음, (4) 모델 로드 실패 시 서비스 기동 자체가 실패함이 테스트로 확인되면
Phase 5 코드는 완료로 간주한다. 실측치는 이후 작성할 `P5_테스트결과서_Inference.md`에
기록한다. 실제 Embedding Service(`localhost:8000`)를 띄워 둔 상태의 수동 E2E 검증은
[[P3_테스트결과서_Training]]/[[P4_테스트결과서_Validation]]과 같은 방식으로
테스트결과서 별도 절에 기록한다.

## 11. 관련 문서/코드

- 요구사항: [[P5_요구사항정의서_Inference]]
- 상위 설계: [[Architecture_Design]] 2절(모듈 구조), 6절(`Dockerfile.inference`)
- 참조 패턴: [[P4_설계서_Validation]](Protocol 재사용, fake 기반 통합 테스트),
  [[project_inference_no_qdrant_registration]](Embedding Service만 호출, AIPro+ 미사용)
- 공통 모듈: `domain/interfaces.py`(`TextClassifier` 신규, `EmbeddingClient`/`Classifier`
  갱신 없음), `exceptions.py`(`EmbeddingServerError`/`ModelNotFoundError`, 갱신 없음),
  `preprocessing/text_cleaner.py`(갱신 없음, 재사용), `evaluation/metrics.py`(갱신 없음,
  재사용)
- 관련 코드(신규): `src/embedding_lr/inference/{embedding_lr_classifier,predictor,api}.py`,
  `src/embedding_lr/cli/run_inference_server.py`, `docker/Dockerfile.inference`
- 관련 코드(갱신): `src/embedding_lr/domain/interfaces.py`,
  `src/embedding_lr/domain/models.py`, `src/embedding_lr/config.py`,
  `pyproject.toml`(`inference` extras)
- 테스트: `tests/unit/test_domain_models.py`(갱신), `tests/integration/test_{embedding_lr_classifier,
  predictor,api,run_inference_server}.py`(신규)
- 신규 의존성: `fastapi`, `uvicorn[standard]`(`inference` extras, `pyproject.toml`)
