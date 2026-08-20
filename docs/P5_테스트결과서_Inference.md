# 테스트결과서 — Phase 5 추론 (Inference)

[[P5_설계서_Inference]]에 따라 구현한 `domain/interfaces.py`(갱신, `TextClassifier`),
`domain/models.py`(갱신, `ClassifyRequest`/`ClassifyResponse`), `config.py`(갱신,
`model_path`/`inference_host`/`inference_port`), `inference/{embedding_lr_classifier,
predictor,api}.py`(신규), `cli/run_inference_server.py`(신규)의 테스트 결과를 기록한다.
[[CLAUDE.md]] 6절에 따라 **모든 실행은 호스트가 아니라 Docker 컨테이너 내부에서**
이뤄졌다.

이 문서는 성격이 다른 두 검증을 구분해서 기록한다.

- **1~3절 — 기능 테스트(자동화 `pytest` 스위트)**: `tests/unit`, `tests/integration`
  아래 자동화 테스트. fake `TextClassifier`/`EmbeddingClient`/`Classifier`와 FastAPI
  `TestClient`로 로직/분기를 빠르고 결정적으로 검증한다 — 실제 AIPro+/Embedding Service를
  호출하지 않는다.
- **4절 — 통합 테스트(실서비스 End-to-End)**: 자동화 스위트에는 포함되지 않은, 실제
  Embedding Service(`localhost:8000`)를 띄운 상태에서 컨테이너를 기동해 `POST /classify`를
  직접 호출한 1회성 검증이다. 모델은 [[P4_테스트결과서_Validation]] 4절에서 실데이터로
  재현했던 `model.pkl`을 그대로 재사용했다.

## 1. 기능 테스트 — 실행 방법

```bash
docker build -f docker/Dockerfile.inference \
  --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy \
  --build-arg no_proxy=$no_proxy \
  -t embedding-lr-inference:phase5 .

# Phase 5 범위만
docker run --rm embedding-lr-inference:phase5 python -m pytest -q \
  tests/unit/test_domain_models.py \
  tests/integration/test_embedding_lr_classifier.py tests/integration/test_predictor.py \
  tests/integration/test_api.py tests/integration/test_run_inference_server.py \
  tests/integration/test_config.py tests/integration/test_aipro_client.py \
  tests/integration/test_embedding_server_client.py tests/integration/test_logging_config.py \
  tests/integration/test_run_context.py \
  --cov=embedding_lr.inference --cov=embedding_lr.domain --cov=embedding_lr.cli.run_inference_server \
  --cov=embedding_lr.config --cov-report=term-missing

# 프로젝트 전체(회귀 확인)
docker run --rm embedding-lr-inference:phase5 python -m pytest -q \
  --cov=embedding_lr --cov-report=term-missing
```

## 2. 기능 테스트 결과 요약

```
Phase 5 범위: 71 passed in 5.22s
프로젝트 전체: 168 passed in 6.84s (Phase 4 시점 150건 대비 +18, 신규 케이스 수와 정확히 일치)
```

Phase 5 범위(A+B 등급 전체, `domain` 패키지 전체 포함) 커버리지: 187 lines 중 186 covered
= **99%**(프로젝트 목표 ≥ 80%, 등급 A 목표 ≥ 90%/등급 B 목표 ≥ 70% 모두 충족). 프로젝트
전체 커버리지는 792 lines 중 782 covered = **99%**(회귀 없음 — Phase 1~4까지의 모듈
커버리지 변동 없음, `config.py`에 필드 3개 추가되었지만 100% 유지).

| 모듈 | 등급([[CLAUDE.md]] 2절) | 목표 커버리지 | 실측 커버리지 | 비고 |
|---|---|---|---|---|
| `domain/interfaces.py`(`TextClassifier` 추가분) | — | — | 100% (9 lines) | Protocol 정의만 — 실행 코드 없음, 커버리지 측정 대상 아니나 100% 표시(다른 Protocol과 동일) |
| `domain/models.py`(`ClassifyRequest`/`ClassifyResponse` 추가분 포함) | A | ≥ 90% | 100% (104 lines) | 순수 래퍼 모델 — 빈 리스트/정상 리스트 생성, 필수 필드 누락 |
| `config.py`(`model_path`/`inference_host`/`inference_port` 추가분 포함) | — | — | 100% (16 lines) | pydantic-settings 필드 선언만 |
| `inference/embedding_lr_classifier.py` | B | ≥ 70% | 100% (10 lines) | fake `EmbeddingClient`/`Classifier` — 빈 리스트 시 호출 없음, 정상 입력 시 각 1회 호출 |
| `inference/predictor.py` | B | ≥ 70% | 100% (12 lines) | fake `TextClassifier` — 빈 리스트 시 미호출, `response` 필드 무영향, 1회 호출, 순서 대응 |
| `inference/api.py` | B | ≥ 70% | 100% (19 lines) | FastAPI `TestClient` + fake `TextClassifier` — 정상 응답, 빈 리스트, `EmbeddingServerError`→503, `/health` |
| `cli/run_inference_server.py` | B | ≥ 70% | 94% (17 lines, 1 miss: L37 `if __name__ == "__main__"`) | fixture 모델 `.pkl` + monkeypatch `uvicorn.run` — host/port/app 검증, 모델 부재 시 `ModelNotFoundError`로 `uvicorn.run` 미호출 확인 |

`if __name__ == "__main__":` 1줄은 다른 CLI 스크립트([[P3_테스트결과서_Training]] 2절)와
동일하게 표준 관용구라 커버리지 측정에서 실질적 의미가 없다.

## 3. 기능 테스트 케이스 상세

| 파일 | 케이스 수 | 확인 내용 |
|---|---|---|
| `tests/unit/test_domain_models.py`(신규 케이스만) | 6 | `ClassifyRequest` 리스트 생성/빈 리스트 허용/필수 필드 누락, `ClassifyResponse` 리스트 생성/빈 리스트 허용/필수 필드 누락 |
| `tests/integration/test_embedding_lr_classifier.py`(신규) | 2 | 빈 리스트 시 `embed()`/`predict_proba()` 미호출, 정상 입력 시 `embed()`→`predict_proba()` 순서로 각 1회 호출 및 결과 전달 확인 |
| `tests/integration/test_predictor.py`(신규) | 4 | 빈 리스트 시 `classifier.classify()` 미호출, 정제된 질의로 `classify()` 1회 호출 + 판정 결과 확인, `response` 필드가 결과에 영향 없음(동일 질의·다른 응답 → 동일 결과), 결과가 요청 순서와 대응 |
| `tests/integration/test_api.py`(신규) | 4 | `/classify` 정상 응답, 빈 리스트 요청 시 빈 결과(및 `classify()` 미호출), `EmbeddingServerError` 발생 시 HTTP 503, `/health` 200 |
| `tests/integration/test_run_inference_server.py`(신규) | 2 | `uvicorn.run`이 설정된 host/port/`FastAPI` 앱으로 호출됨, 모델 파일 부재 시 `ModelNotFoundError`가 발생하고 `uvicorn.run`은 호출되지 않음(기동 자체 실패) |
| `tests/integration/test_config.py`(갱신, 기존 케이스에 검증 추가) | 0(신규 케이스 없음) | 기존 4개 테스트에 `model_path`/`inference_host`/`inference_port` 필드 검증을 추가(신규 필드 로드, 기본값, 필수 필드 누락 시 `ValidationError`) |
| `tests/integration/test_{aipro_client,embedding_server_client,logging_config,run_context}.py`(갱신) | 0(신규 케이스 없음) | `Settings()` 생성자 호출부에 신규 필수 필드 `model_path` 값을 추가해 기존 케이스가 계속 통과하도록 함(회귀 방지, 신규 검증 로직 아님) |

이 4개 신규 통합 테스트 파일은 모두 fake 의존성(HTTP/모델 호출 없음)을 쓴다 — 목적이
"조립·분기 로직이 맞는가"이지 "실제 임베딩·모델이 정확한가"가 아니기 때문이다. 후자는
4절에서 별도로 검증했다.

## 4. 통합 테스트 — 실서비스 End-to-End 상세 결과

### 4.1 실행 방법

```bash
docker run --rm -d --name inference-e2e-test \
  --network host \
  -v "<phase4_out_dir>/model.pkl:/app/model.pkl:ro" \
  -e AIPRO_BASE_URL=http://localhost:28000 -e AIPRO_API_TOKEN=dummy \
  -e EMBEDDING_SERVER_BASE_URL=http://localhost:8000 \
  -e MODEL_DIR=/app -e MODEL_PATH=/app/model.pkl \
  -e INFERENCE_HOST=0.0.0.0 -e INFERENCE_PORT=8090 \
  embedding-lr-inference:phase5
```

`<phase4_out_dir>/model.pkl`은 [[P4_테스트결과서_Validation]] 4.1절에서 실데이터
(`data/v0.2/{train,test}_vectors.parquet`)로 재현했던 바로 그 모델이다 — 별도 재학습 없이
그대로 재사용했다. `--network host`로 컨테이너가 호스트의 `localhost:8000`(실제 기동 중인
Embedding Service)에 직접 접근하게 했다. `AIPRO_*`는 `Settings()` 필수 필드 통과용으로만
존재하며, 이 서비스는 실제로 AIPro+를 호출하지 않는다.

### 4.2 헬스체크 및 정상 분류 확인

```bash
curl http://localhost:8090/health
# {"status":"ok"}

curl -X POST http://localhost:8090/classify -H "Content-Type: application/json" -d '{
  "items": [
    {"query": "쿠버네티스 파드가 CrashLoopBackOff 상태인데 어떻게 확인하나요?", "response": ""},
    {"query": "오늘 점심 뭐 먹을지 고민되는데 추천해줘", "response": ""},
    {"query": "조선시대 왕 순서 알려줘", "response": ""},
    {"query": "재미있는 영화 추천해줘", "response": ""},
    {"query": "asdkfj alksjdf 123123", "response": ""}
  ]
}'
```

5개 항목(IT/DAILY/KNOWLEDGE/CREATIVE/ANOMALY 각 1건씩, 실제 서비스에 들어올 법한 질의로
직접 작성) 모두 기대한 카테고리로 정확히 분류됐다:

| 질의(요약) | 기대 카테고리 | `predicted_category` | `final_verdict` | 최고 확률 |
|---|---|---|---|---|
| 쿠버네티스 파드 CrashLoopBackOff | IT | IT | IT | 0.9732 |
| 점심 메뉴 추천 | DAILY | DAILY | NON_IT | 0.9709 |
| 조선시대 왕 순서 | KNOWLEDGE | KNOWLEDGE | NON_IT | 0.5665 |
| 영화 추천 | CREATIVE | CREATIVE | NON_IT | 0.9141 |
| 무의미 문자열 | ANOMALY | ANOMALY | NON_IT | 0.8856 |

KNOWLEDGE 항목의 확신도(0.5665)가 다른 항목보다 낮게 나왔는데, 2위 확률이 CREATIVE
(0.2614)였다 — "역사 상식"과 "창작/교양" 경계가 임베딩 공간에서 상대적으로 가깝다는
것을 보여주는 사례로, [[P4_테스트결과서_Validation]]에서 이미 관찰된 클래스 간 근접성
(DAILY↔CREATIVE 오분류 3건)과 같은 계열의 현상이다. 5건 모두 최종 판정(`predicted_category`)
자체는 정확했다.

### 4.3 `response` 필드 무시 확인

동일 질의("쿠버네티스 파드가 CrashLoopBackOff...")에 `response`만 완전히 다른 텍스트로
바꿔 재요청한 결과, `IT` 확률이 `0.9731881269895601` → `0.9731881269895599`로 마지막
자릿수만 부동소수점 오차 수준으로 다르고 사실상 동일했다 — `response`가 분류 결과에
영향을 주지 않음을 실서비스로도 확인했다([[P5_요구사항정의서_Inference]] 7절 완료 기준).

### 4.4 빈 리스트 요청 확인

```bash
curl -X POST http://localhost:8090/classify -H "Content-Type: application/json" -d '{"items": []}'
# {"results":[]}
```

### 4.5 정리

검증 후 `docker stop`/`docker rm`으로 테스트 컨테이너를 정리했다. 실제 Embedding
Service·AIPro+에는 어떤 상태 변경도 남기지 않았다(이 서비스는 애초에 두 서비스 중
Embedding Service만 읽기 전용으로 호출한다).

## 5. 완료 기준([[P5_요구사항정의서_Inference]] 7절) 충족 확인

- [x] `POST /classify`에 리스트 요청을 보내면 순서가 대응하는 리스트 응답을 받음(빈
      리스트 포함) — 3절(`TestPredict::test_results_correspond_to_items_in_order`), 4.2/4.4절
- [x] 요청의 `response` 필드가 분류 결과에 영향을 주지 않음 확인 — 3절, 4.3절(실서비스로 재확인)
- [x] `embed()`/`predict_proba()`(→ 현재는 `classifier.classify()`)가 요청 1건당 각각
      정확히 1회만 호출됨 — 3절(`TestEmbedAndPredict`/`TestPredict` call count 검증)
- [x] `GET /health`가 모델 로드 성공 후 200 OK 반환 — 3절, 4.2절
- [x] 모델 로드 실패 시 서비스 기동 자체가 실패함 — 3절(`test_raises_when_model_file_missing`)
- [x] `predictor.py`가 `TextClassifier` Protocol에만 의존하고 httpx/sklearn을 직접
      import하지 않음 확인 — 코드 검토(4절 참고), `EmbeddingClient`/`Classifier`를 동시에
      아는 곳은 `embedding_lr_classifier.py` 하나뿐임을 유지
- [x] Docker 컨테이너(`Dockerfile.inference`) 내부에서 서비스 기동+테스트 실행 확인 — 1절/4.1절
- [x] 등급 A(`domain/models.py` 추가분) 커버리지 ≥90%, 등급 B(`embedding_lr_classifier.py`/
      `predictor.py`/`api.py`) 커버리지 ≥70% 확인 — 2절 표(실측 94~100%)

## 6. 재작업 내역

기능 테스트·통합 테스트 모두 첫 구현에서 바로 통과했다 — 재작업 없음. 다만 `config.py`에
`model_path`를 **필수** 필드로 추가하면서, 이미 존재하던 `Settings()` 생성자 호출부
9곳(`test_run_phase{1,1_5,2,3,4}.py`, `test_config.py`, `test_aipro_client.py`,
`test_embedding_server_client.py`, `test_logging_config.py`, `test_run_context.py`)이
전부 `model_path` 누락으로 `ValidationError`를 던지게 되어, 회귀 방지 차원에서 각
호출부에 `model_path`(또는 `MODEL_PATH` 환경변수) 값을 추가했다 — 새 버그가 아니라
"환경마다 달라지는 필수 설정을 늘렸을 때 기존 테스트 fixture를 함께 갱신해야 한다"는
당연한 후속 작업이었다.

## 7. 관련 문서/코드

- 요구사항/설계: [[P5_요구사항정의서_Inference]], [[P5_설계서_Inference]],
  [[Architecture_Design]] 2절/6절
- 코드(신규): `src/embedding_lr/inference/{embedding_lr_classifier,predictor,api}.py`,
  `src/embedding_lr/cli/run_inference_server.py`, `docker/Dockerfile.inference`
- 코드(갱신): `src/embedding_lr/domain/interfaces.py`(`TextClassifier`),
  `src/embedding_lr/domain/models.py`(`ClassifyRequest`/`ClassifyResponse`),
  `src/embedding_lr/config.py`(`model_path`/`inference_host`/`inference_port`),
  `pyproject.toml`(`inference` extras)
- 테스트: `tests/unit/test_domain_models.py`(갱신), `tests/integration/test_{embedding_lr_classifier,
  predictor,api,run_inference_server}.py`(신규), `tests/integration/test_{config,
  aipro_client,embedding_server_client,logging_config,run_context}.py`(갱신, fixture만)
- 실행 이미지: `docker/Dockerfile.inference`(신규, `inference` extras 설치)
