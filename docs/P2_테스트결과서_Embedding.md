# 테스트결과서 — Phase 2 임베딩 변환 (Embedding)

[[P2_설계서_Embedding]]에 따라 구현한 `embedding/{collection,aipro_client,
embedding_server_client,registration,knowledge_writer,pipeline}`, `cli/run_phase2`의
자동화 테스트 실행 결과를 기록한다. [[CLAUDE.md]] 6절에 따라 **테스트는 호스트가 아니라
Docker 컨테이너 내부에서만** 실행했다. 실제 AIPro+ 서버는 호출하지 않았다 — HTTP
경계는 전부 `respx`(HTTP 모킹) 또는 fake `VectorStore`/`DataRepository`(Protocol
테스트 더블)로 대체했다([[CLAUDE.md]] 2절 B등급 규칙, [[Architecture_Design]] 7절).

## 1. 실행 방법

```bash
docker build -f docker/Dockerfile.pipeline \
  --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy \
  --build-arg no_proxy=$no_proxy \
  -t embedding-lr-pipeline:dev .

# Phase 2 범위만
docker run --rm embedding-lr-pipeline:dev python -m pytest -q \
  tests/unit/test_collection.py \
  tests/integration/test_aipro_client.py tests/integration/test_embedding_server_client.py \
  tests/integration/test_registration.py tests/integration/test_knowledge_writer.py \
  tests/integration/test_pipeline.py tests/integration/test_run_phase2.py \
  --cov=embedding_lr.embedding --cov=embedding_lr.cli.run_phase2 --cov-report=term-missing

# 프로젝트 전체(회귀 확인)
docker run --rm embedding-lr-pipeline:dev python -m pytest -q --cov=embedding_lr --cov-report=term-missing
```

## 2. 결과 요약

```
Phase 2 범위: 35 passed in 2.42s
프로젝트 전체: 98 passed in 2.38s
```

Phase 2 범위(A+B 등급 전체) 커버리지: 162 lines 중 157 covered = **97%**(프로젝트 목표
≥ 80% 충족). 프로젝트 전체 커버리지는 506 lines 중 499 covered = **99%**.

| 모듈 | 등급([[CLAUDE.md]] 2절) | 목표 커버리지 | 실측 커버리지 | 비고 |
|---|---|---|---|---|
| `embedding/collection.py` | A | ≥ 90% | 100% (16 lines) | 점 치환, 미확인 split/파일명 케이스 포함 |
| `embedding/aipro_client.py` | B | ≥ 70% | 92% (49 lines, 4 miss: L88-89, L93-94 — `_post()`의 HTTP 에러/응답 파싱 에러 분기. `create_domain`/`create_collection`은 happy path만 테스트, 그 두 메서드가 실패하는 케이스는 아직 커버 안 됨) | `get_knowledge`/`upsert` happy path+에러, `list_domains`/`create_domain`/`list_collections`/`create_collection`은 `test_run_phase2.py`의 E2E 경로에서 happy path로 간접 커버 |
| `embedding/embedding_server_client.py` | B | ≥ 70% | 100% (18 lines) | Phase 5 전용이라 Phase 2 파이프라인에서는 직접 쓰이지 않지만, 완결성을 위해 자체 테스트로 100% 확보 |
| `embedding/registration.py` | B | ≥ 70% | 100% (12 lines) | fake `VectorStore` — 존재 시 재생성 안 함/미존재 시 생성 각 2케이스 |
| `embedding/knowledge_writer.py` | B | ≥ 70% | 100% (12 lines) | fake `VectorStore` — 매핑 정확성, 배치 1회 호출, category 누락 에러 |
| `embedding/pipeline.py` | B | ≥ 70% | 100% (32 lines) | fake `DataRepository`/`VectorStore` — 등록+저장, 스킵, 출력 존재 시 실패, 재등록 후 불일치 시 실패 4케이스 |
| `cli/run_phase2.py` | B | ≥ 70% | 96% (23 lines, 1 miss: L40 `if __name__ == "__main__"`) | respx로 AIPro+ 전체 흐름(domains/collections/knowledge) 모킹 — happy path + 상태 파일 |

`if __name__ == "__main__":` 1줄은 다른 CLI 스크립트([[P1_테스트결과서_DataPreparation]]
2절)와 동일하게 표준 관용구라 커버리지 측정에서 실질적 의미가 없다.

## 3. 테스트 케이스 상세

| 파일 | 케이스 수 | 확인 내용 |
|---|---|---|
| `tests/unit/test_collection.py` | 8 | `collection_name()` 정상 생성(점 치환 포함), 알 수 없는 split 거부, `extract_version_and_split()` 정상 추출(train/test/val), 알 수 없는 파일명 거부 |
| `tests/integration/test_aipro_client.py` | 9 | `get_knowledge` happy path/쿼리 파라미터 전달/빈 결과/HTTP 에러/응답 파싱 실패(5), `upsert` 레코드별 개별 POST(bulk-upload 아님)/페이로드 필드/HTTP 에러(3), `close()`(1) |
| `tests/integration/test_embedding_server_client.py` | 5 | `embed()` happy path, 요청 바디 형식, HTTP 에러, 응답 파싱 실패, `close()` |
| `tests/integration/test_registration.py` | 4 | `ensure_domain`/`ensure_collection` 각각 기존 존재 시 재생성 안 함/미존재 시 생성 |
| `tests/integration/test_knowledge_writer.py` | 3 | query→content/category→source 매핑, 여러 레코드를 한 번의 `upsert` 호출로 처리, category 누락 시 레코드 인덱스 포함 `DataValidationError` |
| `tests/integration/test_pipeline.py` | 4 | 콜렉션 비어있을 때 등록+parquet 저장, 건수 일치 시 재등록 스킵(기존 임베딩 값 그대로 사용 확인), 출력 경로 기존 존재 시 실패, 재등록 후에도 건수 부족 시 실패 |
| `tests/integration/test_run_phase2.py` | 2 | CLI E2E(respx로 AIPro+ 전체 모킹) — 등록+parquet 저장 happy path, `status/phase2_*.json`에 `succeeded` 기록 |

## 4. 알려진 갭 (재작업 대상 아님, 다음 작업 시 참고)

- `AIProClient._post()`의 두 에러 분기(HTTP 에러, 응답 파싱 실패)는 `create_domain`/
  `create_collection`을 통해서만 도달하는데, 두 메서드 모두 happy path만 테스트되어
  있어 92%에 머문다(B등급 목표 70%는 충족). 필요 시 `test_aipro_client.py`에
  `TestCreateDomain`/`TestCreateCollection`(HTTP 에러 케이스 포함)을 추가하면 100%까지
  끌어올릴 수 있다.
- `embedding_server_client.py`(Phase 5용)는 이번 커밋에서 함께 구현·테스트되었지만,
  Phase 2 파이프라인 어디에서도 호출되지 않는다 — 실제 사용은 `inference/predictor.py`
  (아직 미구현) 몫이다.

## 5. 재작업 내역

- 없음 — 최초 구현에서 이미 그린 상태(35/35, 98/98)였고, 이 문서 작성 과정의 재검증
  실행에서도 동일하게 재현되어 추가 수정 없음.

## 6. 관련 문서/코드

- 요구사항/설계: [[Scope_Definition]] 2.1절, [[Architecture_Design]] 2절/4절,
  [[P2_설계서_Embedding]]
- 공통 모듈 변경 배경: [[P0_테스트결과서_Common_v2]]
- 코드: `src/embedding_lr/embedding/{collection,aipro_client,embedding_server_client,
  registration,knowledge_writer,pipeline}.py`, `src/embedding_lr/cli/run_phase2.py`
- 테스트: `tests/unit/test_collection.py`, `tests/integration/test_{aipro_client,
  embedding_server_client,registration,knowledge_writer,pipeline,run_phase2}.py`
- 실행 이미지: `docker/Dockerfile.pipeline`
