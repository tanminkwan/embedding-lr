# 테스트결과서 — Common (공통 모듈) v2

[[P0_테스트결과서_Common]](v1)의 후속 버전. `domain/models.py`/`domain/interfaces.py`가
[[P0_설계서_Common]] 갱신(`EmbeddingVector` → `KnowledgeRecord`, `KnowledgeItem` 추가,
`VectorStore.upsert`/`get_knowledge` 시그니처 확정)에 따라 바뀌어 재실행한 결과다.
v1은 히스토리 보존을 위해 그대로 둔다([[CLAUDE.md]] 7절 "입출력 보존"과 동일 취지).
[[CLAUDE.md]] 6절에 따라 **테스트는 호스트가 아니라 Docker 컨테이너 내부에서만 실행**했다.

## 0. 변경 배경

Phase 2 설계 중 AIPro+의 실제 `POST /api/rag/knowledge` 요청 스키마(`KnowledgeCreate`)를
확인한 결과 `content`(텍스트)만 받고 벡터 필드가 없다는 사실이 드러났다 — AIPro+가
content로부터 내부적으로 임베딩을 계산해 저장한다(사용자 확인, 2026-08-19: "content
받아서 aipro plus 가 vector 생성해서 등록하는 거야"). 이에 따라:

- `EmbeddingVector`(vector+category) 모델 삭제 — 어디에도 쓰이지 않게 됨(Phase 2 파이프라인은
  `embed()`를 호출하지 않고, `VectorStore.upsert()`도 벡터를 받지 않음).
- `KnowledgeRecord`(content/extended_content/source) 모델 신설 — `VectorStore.upsert()`의
  실제 입력 타입.
- `VectorStore.upsert(records: list[EmbeddingVector], ...)` →
  `VectorStore.upsert(records: list[KnowledgeRecord], ...)`로 시그니처 정정.

## 1. 실행 방법

```bash
docker build -f docker/Dockerfile.pipeline \
  --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy \
  --build-arg no_proxy=$no_proxy \
  -t embedding-lr-pipeline:dev .

docker run --rm embedding-lr-pipeline:dev python -m pytest -q \
  tests/unit/test_domain_models.py \
  tests/integration/test_config.py tests/integration/test_logging_config.py tests/integration/test_run_context.py \
  --cov=embedding_lr.config --cov=embedding_lr.constants --cov=embedding_lr.domain \
  --cov=embedding_lr.exceptions --cov=embedding_lr.logging_config --cov=embedding_lr.workflow.run_context \
  --cov-report=term-missing
```

- v1과 동일하게 Common 모듈(`config`, `constants`, `domain/*`, `exceptions`,
  `logging_config`, `workflow/run_context`)로 커버리지 범위를 한정했다 — Phase 2용
  `embedding/*`(아직 미완성)는 이 보고서 범위 밖이다.

## 2. 결과 요약

```
24 passed in 0.28s
```

| 모듈 | 등급([[CLAUDE.md]] 2절) | 목표 커버리지 | 실측 커버리지 | 비고 |
|---|---|---|---|---|
| `constants.py` | 해당 없음 | 측정 제외 | 100% (13 lines) | 변경 없음 |
| `config.py` | B | ≥ 70% | 100% | 변경 없음 |
| `domain/models.py` | A | ≥ 90% | 100% (65 lines) | `EmbeddingVector` 삭제, `KnowledgeRecord`/`KnowledgeItem` 추가 |
| `domain/interfaces.py` | 해당 없음 | 측정 제외 | 0% (8 lines) | Protocol 선언만 — `VectorStore.upsert`가 `KnowledgeRecord`를 받도록, `get_knowledge`가 `VectorStore`에 속하도록 정정. 각 구현체 테스트에서 fake/respx로 검증(Phase 2 진행 중) |
| `exceptions.py` | 해당 없음 | 측정 제외 | 이 실행 범위(Common 4개 테스트 파일)에서는 0% (5 lines, `coverage`가 "never imported" 경고) — v1과 동일하게 선언만이라 이 범위에서는 아무 것도 import하지 않음. 프로젝트 전체 테스트(`pytest` 전체 실행, 74 passed)에서는 100% (5/5) — `AIProClientError`/`EmbeddingServerError`는 각각 `tests/integration/test_aipro_client.py`, `test_embedding_server_client.py`가, `DataValidationError`는 `test_jsonl_repository.py`가 실제로 raise/assert해서 확인함 | `EmbeddingServerError` 추가(별도 서비스 예외 분리) |
| `logging_config.py` | B | ≥ 70% | 100% | 변경 없음 |
| `workflow/run_context.py` | B | ≥ 70% | 100% | 변경 없음 |

**A+B 등급 전체 가중 평균**: 148 lines 중 148 covered = **100%** (프로젝트 목표 ≥ 80% 충족).

## 3. 테스트 케이스 상세 (v1 대비 변경분)

| 파일 | 케이스 수 | 확인 내용 |
|---|---|---|
| `tests/unit/test_domain_models.py` | 14(v1: 13) | `TestEmbeddingVector`(3케이스) 제거, `TestKnowledgeRecord`(2케이스: 정상 라벨 허용/미확인 라벨 거부), `TestKnowledgeItem`(2케이스: 임베딩 차원 정상/불일치) 추가 |
| `tests/integration/test_config.py`, `test_logging_config.py`, `test_run_context.py` | 변경 없음 | v1과 동일 |

## 4. 재작업 내역

- 없음 — 최초 실행에 24 passed, 100% 커버리지(A/B 등급 대상).

## 5. 관련 문서/코드

- 설계: [[P0_설계서_Common]] (본 변경 반영), [[P0_설계서_Logging]]
- 이전 버전: [[P0_테스트결과서_Common]] (v1, 보존)
- 코드: `src/embedding_lr/{config,constants,exceptions,logging_config}.py`,
  `src/embedding_lr/domain/{models,interfaces}.py`, `src/embedding_lr/workflow/run_context.py`
- 테스트: `tests/unit/test_domain_models.py`, `tests/integration/test_{config,logging_config,run_context}.py`
- 실행 이미지: `docker/Dockerfile.pipeline`
