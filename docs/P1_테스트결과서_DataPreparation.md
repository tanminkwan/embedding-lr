# 테스트결과서 — Phase 1 데이터 준비 (DataPreparation)

[[P1_설계서_DataPreparation]]에 따라 구현한 `data_generation/{csv_repository,jsonl_repository}`,
`dataset/{combine,split}`, `cli/{run_phase1,run_phase1_5}`, `preprocessing/text_cleaner`의
자동화 테스트 실행 결과와, 실제 CLI를 Docker 컨테이너에서 구동해 확인한 수동(E2E) 검증
결과를 함께 기록한다. [[CLAUDE.md]] 6절에 따라 **테스트/실행 모두 호스트가 아니라 Docker
컨테이너 내부에서만** 수행했다.

## 1. 자동화 테스트 — 실행 방법

```bash
docker build -f docker/Dockerfile.pipeline \
  --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy \
  --build-arg no_proxy=$no_proxy \
  -t embedding-lr-pipeline:test .

docker run --rm embedding-lr-pipeline:test \
  python -m pytest --cov=embedding_lr --cov-report=term-missing
```

- 프록시 build-arg는 현재 개발 환경(사내 프록시 경유) 때문에 필요하며 자동 상속되지
  않는다 — 값은 하드코딩하지 않고 호스트 쉘 환경변수를 그대로 전달한다.

## 2. 자동화 테스트 — 결과 요약

```
62 passed in 0.72s
```

전체(A+B 등급) 커버리지: 324 lines 중 315 covered = **97%**(프로젝트 목표 ≥ 80% 충족).
`domain/interfaces.py`는 [[P0_테스트결과서_Common]]과 동일하게 Protocol 선언만이므로
측정 제외 대상이다.

| 모듈 | 등급([[CLAUDE.md]] 2절) | 목표 커버리지 | 실측 커버리지 | 비고 |
|---|---|---|---|---|
| `data_generation/csv_repository.py` | B | ≥ 70% | 100% (20 lines) | happy path + 필수 컬럼 누락/필드 검증 실패 |
| `data_generation/jsonl_repository.py` | B | ≥ 70% | 100% (32 lines) | happy path + JSON 파싱 실패/필수 키 누락/기존 파일 존재 시 거부 |
| `cli/run_phase1.py` | B | ≥ 70% | 95% (20 lines, 1 miss: L40 `if __name__ == "__main__"`) | 통합 테스트로 CSV→JSONL 변환 happy path 검증 |
| `cli/run_phase1_5.py` | B | ≥ 70% | 97% (35 lines, 1 miss: L64 `if __name__ == "__main__"`) | 통합 테스트로 재조합+분할 happy path 검증 |
| `dataset/combine.py` | A | ≥ 90% | 100% (24 lines) | 클래스별 건수 불일치, (질의,카테고리) 중복, 카테고리 누락 케이스 포함 |
| `dataset/split.py` | A | ≥ 90% | 100% (26 lines) | seed 고정 재현성, 비율 합으로 나누어떨어지지 않는 케이스 포함 |
| `preprocessing/text_cleaner.py` | A | ≥ 90% | 100% (20 lines) | 코드펜스/스택트레이스 라인 제거, 공백 정규화 3개 규칙 |

`if __name__ == "__main__":` 블록 2줄은 CLI 진입점 표준 관용구로, 실제 실행 경로는
아래 3절의 수동 E2E 검증에서 별도로 확인했다.

## 3. 테스트 케이스 상세

| 파일 | 케이스 수 | 확인 내용 |
|---|---|---|
| `tests/integration/test_csv_repository.py` | 5 | 정상 로드, 필수 컬럼 누락 시 줄 번호 포함 에러, 필드 검증 실패(카테고리 미허용값 등), `save()` 호출 시 `NotImplementedError` |
| `tests/integration/test_jsonl_repository.py` | 6 | 정상 로드/저장, 빈 줄 스킵, JSON 파싱 실패 시 줄 번호 포함 에러, 필수 키 누락, 저장 대상 파일 기존 존재 시 거부 |
| `tests/integration/test_run_phase1.py` | 3 | CSV→JSONL 변환 happy path, 산출물 라인 수/내용 일치, 존재하는 출력 경로로 재실행 시 실패 |
| `tests/integration/test_run_phase1_5.py` | 2 | role 9개 로드→재조합→분할→저장 happy path, `DATA_SPLITS`/`SPLIT_FILE_STEMS` 매핑에 따른 파일명 확인 |
| `tests/unit/test_combine.py` | 4 | 정상 조합(1,000건), 클래스별 건수 불일치 시 어떤 클래스가 몇 건인지 메시지 포함 에러, (질의,카테고리) 중복 거부, 카테고리 누락 거부 |
| `tests/unit/test_split.py` | 6 | 3:1:1 비율대로 분할, 동일 seed 재실행 시 바이트 단위 동일 결과, 다른 seed면 결과 상이, 비율 합으로 나누어떨어지지 않는 클래스 거부, 카테고리 누락 거부 |
| `tests/unit/test_text_cleaner.py` | 13 | 코드펜스 구분자 제거(내용 보존), 스택트레이스 라인 제거, 공백 정규화, 조합 케이스 |

## 4. 수동 E2E 검증 — CLI 실제 구동

자동화 테스트는 각 모듈을 단위/통합 수준에서 검증하지만, 실제 CLI 두 단계
(`run_phase1.py` → `run_phase1_5.py`)를 이어서 구동했을 때도 정상 동작하는지는 별도로
Docker 컨테이너에서 수동 검증했다. 커밋된 `data/v0.2/*.jsonl`을 직접 덮어쓰지 않기 위해
스크래치 디렉터리를 볼륨 마운트해 별도 산출물로 재생성한 뒤, 기존 산출물과 비교했다.

### 4.1 실행 방법

```bash
# Phase 1: role별 CSV → JSONL 변환 (9개 파일 각각)
docker run --rm \
  -v "$(pwd)/data:/data:ro" -v "$SCRATCH/roles:/out" -v "$SCRATCH/status:/status" \
  -e AIPRO_BASE_URL=http://localhost:28000 -e AIPRO_API_TOKEN=dummy \
  -e MODEL_DIR=/tmp/models -e STATUS_DIR=/status \
  embedding-lr-pipeline:test \
  python -m embedding_lr.cli.run_phase1 --input /data/v0.1_from.Claude-Cowork/role_0N_*.csv --output /out/role_0N_*.jsonl

# Phase 1.5: 9개 role JSONL → 재조합 + 3:1:1 분할
docker run --rm \
  -v "$SCRATCH/roles:/roles:ro" -v "$SCRATCH/out:/out" -v "$SCRATCH/status:/status" \
  -e AIPRO_BASE_URL=http://localhost:28000 -e AIPRO_API_TOKEN=dummy \
  -e MODEL_DIR=/tmp/models -e STATUS_DIR=/status \
  embedding-lr-pipeline:test \
  python -m embedding_lr.cli.run_phase1_5 --input-dir /roles --output-dir /out
```

- `config.Settings`가 요구하는 필수 필드(`aipro_base_url`, `aipro_api_token`, `model_dir`)는
  실제 AIPro+를 호출하지 않는 Phase 1/1.5 범위 밖 값이므로 더미 값을 환경변수로 주입했다.

### 4.2 결과

| 확인 항목 | 기대값 | 실측값 | 결과 |
|---|---|---|---|
| Phase 1: role 9개 각각 변환 로그(시작/완료) | 예외 없이 9쌍 출력 | 9쌍 모두 정상 출력 | ✅ |
| Phase 1.5: `data.jsonl` 총 건수 | 1,000건 | 1,000건 | ✅ |
| Phase 1.5: 클래스별 건수 | 각 200건 | `IT/DAILY/KNOWLEDGE/CREATIVE/ANOMALY` 각 200건 | ✅ |
| Phase 1.5: `train/test/val.jsonl` 건수 | 600/200/200 | 600/200/200 | ✅ |
| JSON 유효성(전체 산출 파일) | 파싱 실패 0건 | 파싱 실패 0건(role 9개 + data/train/test/val) | ✅ |
| 재현성 (동일 입력·동일 seed) | 기존 `data/v0.2/*.jsonl`과 바이트 단위 동일 | `diff` 결과 차이 없음(`data.jsonl`/`train.jsonl`/`test.jsonl`/`val.jsonl` 4개 파일) | ✅ |
| 입출력 보존(재실행 시 덮어쓰기 금지) | 기존 출력 존재 시 실패 | `DataValidationError: /out/data.jsonl 이미 존재 — 덮어쓰기 금지` 발생, 종료 코드 1 | ✅ |

[[P1_DataPreparation_Requirements]] 7절의 완료 기준(클래스당 200건, 3:1:1 분할, JSON 유효성,
재현성)을 실제 CLI 구동으로도 재확인했다. 검증에 사용한 스크래치 디렉터리는 세션 종료
전 삭제했으며, 리포지토리 내 `data/v0.2/`나 다른 파일은 변경하지 않았다.

## 5. 재작업 내역

- 없음 — 최초 구현 및 실제 파이프라인 실행(원본 CSV 이스케이프 오류 수정 포함, [[P1_Data_Preprocessing_Review]]
  5절)에서 이미 62 passed / 97% 커버리지로 그린 상태였고, 이번 문서화 과정의 재검증(자동화
  테스트 재실행 + CLI 수동 E2E)에서도 동일하게 재현되어 추가 수정 없음.

## 6. 관련 문서/코드

- 요구사항: [[P1_DataPreparation_Requirements]]
- 설계: [[P1_설계서_DataPreparation]]
- 코드: `src/embedding_lr/data_generation/{csv_repository,jsonl_repository}.py`,
  `src/embedding_lr/dataset/{combine,split}.py`,
  `src/embedding_lr/cli/{run_phase1,run_phase1_5}.py`,
  `src/embedding_lr/preprocessing/text_cleaner.py`
- 테스트: `tests/unit/test_{combine,split,text_cleaner}.py`,
  `tests/integration/test_{csv_repository,jsonl_repository,run_phase1,run_phase1_5}.py`
- 실행 이미지: `docker/Dockerfile.pipeline`
- 산출물: `data/v0.2/{role_01~09,data,train,test,val}.jsonl`
