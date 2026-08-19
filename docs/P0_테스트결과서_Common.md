# 테스트결과서 — Common (공통 모듈)

[[P0_설계서_Common]]에 따라 구현한 공통 모듈(`config`, `constants`, `domain/models`,
`domain/interfaces`, `exceptions`, `workflow/run_context`, `logging_config`)의 테스트
실행 결과. [[CLAUDE.md]] 6절에 따라 **테스트는 호스트가 아니라 Docker 컨테이너 내부에서만
실행**했다.

## 1. 실행 방법

```bash
docker build -f docker/Dockerfile.pipeline \
  --build-arg HTTP_PROXY=$HTTP_PROXY --build-arg HTTPS_PROXY=$HTTPS_PROXY \
  --build-arg NO_PROXY=$NO_PROXY \
  -t embedding-lr-pipeline:dev .

docker run --rm embedding-lr-pipeline:dev pytest -q --cov=embedding_lr --cov-report=term-missing
```

- 프록시 build-arg는 현재 개발 환경 네트워크 설정 때문에 필요(사내 프록시 경유). 값은
  하드코딩하지 않고 호스트 쉘 환경변수를 그대로 전달한다. 인터넷 접근이 자유로운
  환경에서는 생략 가능.

## 2. 결과 요약

```
23 passed in 0.36s
```

| 모듈 | 등급([[CLAUDE.md]] 2절) | 목표 커버리지 | 실측 커버리지 | 비고 |
|---|---|---|---|---|
| `constants.py` | 해당 없음 | 측정 제외 | 100% (9 lines) | 선언만 — `DOMAIN_NAME`/`DATA_SPLITS` 추가, `HASH_ALGORITHM` 제거 |
| `config.py` | B | ≥ 70% | 100% | `.env` 필수 필드/기본값/오버라이드/누락 에러 케이스 |
| `domain/models.py` | A | ≥ 90% | 100% | `QueryRecord`/`EmbeddingVector`/`PredictionResult` 필드 검증 + 교차 검증(예측 카테고리 ↔ 최종 판정 일치). `EmbeddingVector`는 `source_hash` 제거, `category` 필수화 + 라벨 값 검증 추가 |
| `domain/interfaces.py` | 해당 없음 | 측정 제외 | 0% (7 lines) | Protocol 선언만 — `VectorCache.filter_existing` 제거, `VectorStore.upsert`로 단순화. 각 구현체(추후 Phase 모듈) 테스트에서 fake로 대체해 검증 예정 |
| `exceptions.py` | 해당 없음 | 측정 제외 | 0% (4 lines) | 예외 클래스 선언만 — `CacheLookupError` 제거 |
| `logging_config.py` | B | ≥ 70% | 100% | JSON 스키마 필드([[P0_설계서_Logging]] 3절), `extra` 중첩, `LOG_LEVEL` 필터링 |
| `workflow/run_context.py` | B | ≥ 70% | 100% | `run_id` 포맷, 성공/실패 경로 상태 파일 기록 + 예외 재전파 |

**A+B 등급 전체 가중 평균**: 123 lines 중 123 covered = **100%** (프로젝트 목표 ≥ 80% 충족).
등급 "해당 없음"(선언만) 20 lines(`constants.py` 9 + `domain/interfaces.py` 7 +
`exceptions.py` 4)는 [[P0_설계서_Common]] 7절 방침대로 커버리지 측정에서 제외했다.

## 3. 테스트 케이스 상세

| 파일 | 케이스 수 | 확인 내용 |
|---|---|---|
| `tests/unit/test_domain_models.py` | 13 | 정상값 허용, 알 수 없는 `category`/`predicted_category`/`final_verdict` 거부, 벡터 차원 불일치 거부, `EmbeddingVector.category`가 알 수 없는 라벨이면 거부, `probabilities` 키 누락/초과 거부, `predicted_category`-`final_verdict` 불일치 거부 |
| `tests/integration/test_config.py` | 4 | 필수 필드 로딩, 기본값 적용, 환경변수로 기본값 오버라이드, 필수 필드 누락 시 에러 |
| `tests/integration/test_logging_config.py` | 3 | JSON 스키마 필수 필드 포함, `extra` 필드 객체로 중첩, `LOG_LEVEL` 이하 레벨 필터링 |
| `tests/integration/test_run_context.py` | 3 | `run_id` 포맷 정규식, 성공 경로 상태 파일(`succeeded`), 실패 경로 상태 파일(`failed` + `error`) 및 예외 재전파 |

## 4. 재작업 내역

- `python-json-logger` 4.x에서 `pythonjsonlogger.jsonlogger`가 `pythonjsonlogger.json`으로
  이동되며 `DeprecationWarning` 발생 → import 경로를 신규 모듈로 변경.
- 최초 실행 시 `domain/models.py`의 `final_verdict` 값 자체가 `"IT"`/`"NON_IT"` 외의
  임의 문자열(예: `"MAYBE"`)인 경우를 거부하는 검증 분기가 테스트 커버리지에서 누락되어
  98% → 해당 케이스(`test_rejects_unknown_final_verdict_literal`) 추가 후 100%로 보완.
- 임베딩 캐싱 설계 변경(MD5 해시 기반 레코드 중복 판별 제거 → `source`=라벨값 저장)에
  따라 `EmbeddingVector.source_hash`, `VectorCache.filter_existing`, `CacheLookupError`,
  `HASH_ALGORITHM`을 제거하고 `EmbeddingVector.category`를 필수 필드로 변경 —
  `test_rejects_unknown_category`(`TestEmbeddingVector`) 케이스 추가 후 재검증, 22 → 23 passed.

## 5. 관련 문서/코드

- 설계: [[P0_설계서_Common]], [[P0_설계서_Logging]]
- 코드: `src/embedding_lr/{config,constants,exceptions,logging_config}.py`,
  `src/embedding_lr/domain/{models,interfaces}.py`, `src/embedding_lr/workflow/run_context.py`
- 테스트: `tests/unit/test_domain_models.py`, `tests/integration/test_{config,logging_config,run_context}.py`
- 실행 이미지: `docker/Dockerfile.pipeline`
