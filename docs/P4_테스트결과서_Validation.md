# 테스트결과서 — Phase 4 검증 (Validation)

[[P4_설계서_Validation]]에 따라 구현한 `domain/models.py`(갱신), `constants.py`(갱신),
`training/persistence.py`(갱신, `load_search_result` 추가), `evaluation/{metrics,
report}.py`(신규), `cli/run_phase4.py`(신규)의 테스트 결과를 기록한다. [[CLAUDE.md]] 6절에
따라 **모든 실행은 호스트가 아니라 Docker 컨테이너 내부에서** 이뤄졌다. 이 Phase도
Phase 3와 마찬가지로 AIPro+/Embedding Service 등 외부 서비스를 전혀 호출하지 않는다
([[P4_요구사항정의서_Validation]] 3절 Out of Scope) — 입력은 Phase 2/3가 만든 로컬
파일(`val_vectors.parquet`/`model.pkl`/`hyperparams.json`) 뿐이다.

[[P3_테스트결과서_Training]]과 동일하게 두 성격의 검증을 구분해서 기록한다.

- **1~3절 — 기능 테스트(자동화 `pytest` 스위트)**: `tests/unit`, `tests/integration`
  아래 자동화 테스트. fake `Classifier`(고정 확률 반환)와 저차원 합성 fixture로 로직/분기를
  빠르고 결정적으로 검증한다.
- **4절 — 통합 테스트(실데이터 End-to-End)**: Phase 2가 실제로 만든
  `data/v0.2/val_vectors.parquet`(1024D, 5-class, 200건)과, 이 실행을 위해 같은 이미지로
  재현한 Phase 3 실산출물(`model.pkl`/`hyperparams.json`, `data/v0.2/{train,test}_vectors.parquet`
  기반)을 그대로 컨테이너에 마운트해 `cli/run_phase4.main()`을 직접 실행한 1회성 검증이다.

## 1. 기능 테스트 — 실행 방법

```bash
docker build -f docker/Dockerfile.pipeline \
  --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy \
  --build-arg no_proxy=$no_proxy \
  -t embedding-lr-pipeline:phase4 .

# Phase 4 범위만
docker run --rm embedding-lr-pipeline:phase4 python -m pytest -q \
  tests/unit/test_domain_models.py tests/unit/test_metrics.py \
  tests/integration/test_report.py tests/integration/test_run_phase4.py \
  tests/integration/test_persistence.py \
  --cov=embedding_lr.evaluation --cov=embedding_lr.cli.run_phase4 \
  --cov=embedding_lr.domain.models --cov=embedding_lr.training.persistence \
  --cov-report=term-missing

# 프로젝트 전체(회귀 확인)
docker run --rm embedding-lr-pipeline:phase4 python -m pytest -q \
  --cov=embedding_lr --cov-report=term-missing
```

## 2. 기능 테스트 결과 요약

```
Phase 4 범위: 53 passed in 4.63s
프로젝트 전체: 150 passed in 5.33s (Phase 3 시점 121건 대비 +29건, Phase 4 신규분과 정확히 일치)
```

Phase 4 범위(A+B 등급 전체) 커버리지: 206 lines 중 205 covered = **99%**(프로젝트 목표
≥ 80%, 등급 A 목표 ≥ 90%/등급 B 목표 ≥ 70% 모두 충족). 프로젝트 전체 커버리지는 726 lines
중 717 covered = **99%**(회귀 없음 — Phase 3까지의 모듈 커버리지 변동 없음).

| 모듈 | 등급([[CLAUDE.md]] 2절) | 목표 커버리지 | 실측 커버리지 | 비고 |
|---|---|---|---|---|
| `domain/models.py`(`ValidationMetrics`/`GapMetrics`/`TargetCheckResult`/`EvaluationReport` 추가분 포함) | A | ≥ 90% | 100% (100 lines) | 필드 검증만 있는 순수 모델 |
| `evaluation/metrics.py` | A | ≥ 90% | 100% (22 lines) | `probs_to_labels` 동률 tie-break·일부 클래스 누락 케이스, `compute_metrics`/`compute_gap`/`check_targets` 각각 |
| `evaluation/report.py` | B | ≥ 70% | 100% (32 lines) | fake `Classifier`로 `build_report`/`render_markdown`/`save_report`(md·json 각각 기존 존재 시 실패) |
| `training/persistence.py`(`load_search_result` 추가분 포함) | B | ≥ 70% | 100% (21 lines) | 왕복 + 파일 부재 시 `ModelNotFoundError` |
| `cli/run_phase4.py` | B | ≥ 70% | 97% (31 lines, 1 miss: L63 `if __name__ == "__main__"`) | fixture parquet + fixture model.pkl + fixture hyperparams.json + fixture config JSON으로 E2E 실행, 목표 미달 케이스에서도 exit code 0/`succeeded` 상태 확인 |

`if __name__ == "__main__":` 1줄은 다른 CLI 스크립트([[P3_테스트결과서_Training]] 2절)와
동일하게 표준 관용구라 커버리지 측정에서 실질적 의미가 없다.

## 3. 기능 테스트 케이스 상세

| 파일 | 케이스 수 | 확인 내용 |
|---|---|---|
| `tests/unit/test_domain_models.py`(신규 케이스만) | 8 | `ValidationMetrics` 정상 생성/필수 필드 누락, `GapMetrics` warning 플래그/필수 필드 누락, `TargetCheckResult` 전체 달성/필수 필드 누락, `EvaluationReport` 중첩 모델 정상 생성/필수 필드 누락 |
| `tests/unit/test_metrics.py`(신규) | 12 | `probs_to_labels` argmax 반환(1)/동률 시 CLASS_LABELS 순서 tie-break(1)/일부 클래스만 있는 dict 처리(1), `to_binary_labels` IT/NON_IT 매핑(1), `compute_metrics` 완전 일치 시 만점(1)/클래스별 report 키 확인(1)/오분류 시 accuracy 하락(1), `compute_gap` 임계값 이하 무경고(1)/accuracy gap 초과 경고(1)/f1_macro gap 초과 경고(1), `check_targets` 전체 달성(1)/지표별 독립 판정(1) |
| `tests/integration/test_persistence.py`(신규 케이스만) | 2 | `load_search_result` 저장분 왕복, 파일 부재 시 `ModelNotFoundError` |
| `tests/integration/test_report.py`(신규) | 5 | `build_report`가 fake `Classifier` 예측을 지표로 변환(1), `render_markdown`이 주요 섹션(Confusion Matrix/Classification Report) 포함(1), `save_report` md+json 동시 생성(1)/md 기존 존재 시 실패하며 json 미생성(1)/json 기존 존재 시 실패하며 md 미생성(1) |
| `tests/integration/test_run_phase4.py`(신규) | 2 | CLI E2E(fixture parquet+model+search-result+config, 저차원 합성 데이터) — 리포트 md/json 생성 확인, 목표 미달 케이스 포함 `status/phase4_*.json`에 `succeeded` 기록(exit code에 반영하지 않음) 확인 |

`save_report`의 "md만 먼저 쓰고 json에서 실패" 방지 검증(둘 중 하나라도 있으면 아무것도
쓰지 않음)은 설계서 5절이 명시한 요구사항을 그대로 테스트로 옮긴 것이다.

## 4. 통합 테스트 — 실데이터 End-to-End 상세 결과

### 4.1 실행 방법

Phase 4의 입력인 `model.pkl`/`hyperparams.json`은 저장소에 커밋되지 않으므로(산출물
보존 원칙상 out 디렉터리 산물), 먼저 같은 이미지로 Phase 3를 실행해 실데이터 기반
모델을 재현한 뒤 Phase 4를 실행했다.

```bash
mkdir -p <out_dir>/status

# 1) Phase 3 재현 — 실 train/test_vectors.parquet으로 model.pkl/hyperparams.json 생성
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "<out_dir>:/app/out" \
  -e AIPRO_BASE_URL=http://localhost:28000 -e AIPRO_API_TOKEN=dummy \
  -e EMBEDDING_SERVER_BASE_URL=http://localhost:8000 \
  -e MODEL_DIR=/app/out -e STATUS_DIR=/app/out/status \
  embedding-lr-pipeline:phase4 \
  python -m embedding_lr.cli.run_phase3 \
    --train data/v0.2/train_vectors.parquet --test data/v0.2/test_vectors.parquet \
    --model-output /app/out/model.pkl --search-output /app/out/hyperparams.json

# 2) Phase 4 — 실 val_vectors.parquet + 위에서 만든 model.pkl/hyperparams.json으로 검증
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "<out_dir>:/app/out" \
  -e AIPRO_BASE_URL=http://localhost:28000 -e AIPRO_API_TOKEN=dummy \
  -e EMBEDDING_SERVER_BASE_URL=http://localhost:8000 \
  -e MODEL_DIR=/app/out -e STATUS_DIR=/app/out/status \
  embedding-lr-pipeline:phase4 \
  python -m embedding_lr.cli.run_phase4 \
    --val data/v0.2/val_vectors.parquet \
    --model /app/out/model.pkl --search-result /app/out/hyperparams.json \
    --report-md /app/out/eval_report.md --report-json /app/out/eval_report.json
```

`--config`를 생략해 기본값 `config/eval_thresholds_default.json`(`gap_warning_threshold=0.1`,
이미지에 COPY됨)을 그대로 썼다. `AIPRO_*`/`EMBEDDING_SERVER_BASE_URL`은 `Settings()`
필수 필드 통과용으로만 존재하며, 이 CLI는 실제로 그 두 서비스를 호출하지 않는다.

### 4.2 입력 데이터 확인

| 파일 | shape | 클래스별 건수 |
|---|---|---|
| `val_vectors.parquet` | (200, 2) | `IT`/`DAILY`/`KNOWLEDGE`/`CREATIVE`/`ANOMALY` 각 40건 |
| `model.pkl`(재현) | 42,097 bytes | [[P3_테스트결과서_Training]] 4.3절과 동일 조합(`C=10.0`)으로 재현됨 |
| `hyperparams.json`(재현) | 2,227 bytes | best_accuracy=0.995, best_f1_macro≈0.99500 — [[P3_테스트결과서_Training]] 4.4절 실측치와 동일 |

### 4.3 실행 결과 — 상태/타이밍/산출물

| 항목 | 값 |
|---|---|
| `run_id` | `20260820-033250-35cd` |
| 시작 | `2026-08-20T03:32:50.857Z` |
| 종료 | `2026-08-20T03:32:51.122Z` |
| 소요 시간 | 약 0.27초 |
| `status/phase4_*.json` | `"status": "succeeded"`, `"error": null` |
| `eval_report.md`/`.json` | 둘 다 생성, joblib이 아닌 순수 JSON/텍스트라 재로드 이슈 없음 |

### 4.4 검증 지표 실측치

| 지표 | 값 | 목표 | 달성 |
|---|---|---|---|
| 5-class Accuracy | 0.9850 | ≥0.85 | O |
| IT vs NON_IT Accuracy | 1.0000 | ≥0.90 | O |
| F1-macro | 0.98500 | ≥0.85 | O |

**Test-vs-Validation Gap**: Accuracy gap = 0.0100(test 0.995 − val 0.985), F1-macro
gap ≈ 0.0100 — 임계값(0.1) 미만이라 `warning=false`. 목표도 모두 달성했고 gap도 작으므로,
[[P4_요구사항정의서_Validation]] 8절의 해석 가이드상 "과적합도, 데이터 품질 문제도 의심할
근거가 약하다"로 읽힌다 — 이번 라운드는 루프백이 불필요하다는 것이 리포트 기반 판단이다.

**Confusion Matrix(5-class)** — 오분류는 `DAILY→CREATIVE` 2건, `CREATIVE→DAILY` 1건,
총 3건/200건(각 40건씩 IT/KNOWLEDGE/ANOMALY는 완전 정답):

| 실제\예측 | IT | DAILY | KNOWLEDGE | CREATIVE | ANOMALY |
|---|---|---|---|---|---|
| IT | 40 | 0 | 0 | 0 | 0 |
| DAILY | 0 | 38 | 0 | 2 | 0 |
| KNOWLEDGE | 0 | 0 | 40 | 0 | 0 |
| CREATIVE | 0 | 1 | 0 | 39 | 0 |
| ANOMALY | 0 | 0 | 0 | 0 | 40 |

**Confusion Matrix(IT vs NON_IT)** — 이진 집계에서는 오분류 3건 모두 NON_IT 내부
(DAILY↔CREATIVE) 이동이라 완전히 상쇄되어 이진 Accuracy가 1.0000으로 나온다:

| 실제\예측 | IT | NON_IT |
|---|---|---|
| IT | 40 | 0 |
| NON_IT | 0 | 160 |

전체 리포트 원문은 `eval_report.md`/`.json`(1회성 실행 산출물, 저장소에 커밋되지 않음)에
있다.

### 4.5 val set 격리 확인

`data/v0.2/val_vectors.parquet`의 mtime을 Phase 4 실행 전후로 비교해 **변경되지
않았음**을 확인했다(`1787133729` → `1787133729`, [[P3_테스트결과서_Training]] 4.6절이
Phase 3에서 확인한 값과 동일) — Phase 2 이후 이 파일이 어떤 Phase에서도 쓰기 대상이 된
적이 없음을 재확인했다.

## 5. 완료 기준([[P4_요구사항정의서_Validation]] 7절) 충족 확인

- [x] `val_vectors.parquet`(200건)에 대해 5-class Accuracy/IT-NON_IT Accuracy/F1-macro/
      Confusion Matrix/Classification Report가 모두 산출됨 — 4.4절
- [x] `hyperparams.json`의 test 성적과 val 성적 간 gap(Accuracy, F1-macro)이 리포트에
      수치로 포함됨 — 4.4절
- [x] [[Scope_Definition]] 4.4절 목표치 3가지 각각의 달성/미달 여부가 리포트에 명시됨 — 4.4절
- [x] `eval_report_<ver>.md`와 `.json`이 동일 내용으로 생성되고, 재실행 시 기존 파일을
      덮어쓰지 않음 — 3절(`TestSaveReport`), 4.3절
- [x] Phase 4 코드가 `val_vectors.parquet`을 어떤 학습·재학습에도 사용하지 않음(읽기 전용)
      확인 — 4.5절
- [x] `Classifier` Protocol(`predict_proba`)에만 의존하고 sklearn을 직접 import하지 않음
      확인 — `evaluation/report.py`는 sklearn을 import하지 않으며, `evaluation/metrics.py`
      내부 지표 계산만 `sklearn.metrics`를 사용(요구사항 7절 예외 조항과 일치)
- [x] Docker 컨테이너 내부에서 검증 스크립트+테스트 실행 확인 — 1절/4.1절
- [x] 등급 A(`metrics.py`) 커버리지 ≥90%, 등급 B(`report.py`/`run_phase4.py`) 커버리지
      ≥70% 확인 — 2절 표(실측 97~100%)

## 6. 재작업 내역

기능 테스트(합성 fixture) 단계에서 1건의 버그를 발견해 즉시 수정했다 — 실데이터
통합 테스트(4절)까지 갈 필요 없이 자동화 스위트 안에서 잡혔다.

- **`probs_to_labels`가 일부 클래스만 학습된 모델에서 `KeyError`**: 최초 구현은
  `max(CLASS_LABELS, key=lambda label: row[label])`로 CLASS_LABELS 5개 전부가 확률
  dict에 있다고 가정했다. `tests/integration/test_run_phase4.py`의 fixture 모델이
  ([[P3_테스트결과서_Training]]의 기존 fixture 관례를 따라) IT/DAILY 2-class만으로
  학습되어 `predict_proba()`가 2개 키만 반환하자 `KeyError: 'KNOWLEDGE'`가 발생했다.
  실제 Phase 3 파이프라인은 5-class 전체(120건씩)로 학습하므로 운영 경로에서는
  드러나지 않았을 결함이지만, "모델이 일부 클래스만 학습했을 수 있다"는 `Classifier`
  Protocol 계약상 일반적인 가능성을 방어하지 않은 설계 결함이었다. `row[label]` 순회
  전에 `CLASS_LABELS`를 `row`에 실제로 있는 키로 먼저 필터링하도록 수정했고
  (`if label in row`), `tests/unit/test_metrics.py::TestProbsToLabels::
  test_handles_dicts_missing_some_class_labels`로 회귀 방지 케이스를 추가했다.

## 7. 관련 문서/코드

- 요구사항/설계: [[P4_요구사항정의서_Validation]], [[P4_설계서_Validation]],
  [[Architecture_Design]] 2절
- 코드: `src/embedding_lr/evaluation/{metrics,report}.py`(신규),
  `src/embedding_lr/cli/run_phase4.py`(신규), `src/embedding_lr/domain/models.py`(갱신),
  `src/embedding_lr/constants.py`(갱신), `src/embedding_lr/training/persistence.py`(갱신),
  `config/eval_thresholds_default.json`(신규)
- 테스트: `tests/unit/test_domain_models.py`(갱신), `tests/unit/test_metrics.py`(신규),
  `tests/integration/test_{report,run_phase4}.py`(신규),
  `tests/integration/test_persistence.py`(갱신)
- 실행 이미지: `docker/Dockerfile.pipeline`(변경 없음 — `config/`, `src/`, `tests/`
  COPY 범위가 이미 신규 파일을 포함)
