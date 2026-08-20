# P4_요구사항정의서_Validation — Phase 4 요구사항 정의서

[[Scope_Definition]] 4.4절(평가 지표)·4.5절(검증 실패 시 루프백)과 [[Architecture_Design]]
1절/2절(`evaluation/metrics.py`, `evaluation/report.py`, `cli/run_phase4.py` earmark)을
Phase 4 단위로 구체화한 요구사항 정의서. [[CLAUDE.md]] 3절 "작업 진행 순서"의 1단계
산출물이며, 다음 산출물은 `P4_설계서_Validation.md`(설계서)에서 다룬다.

Phase 3와 마찬가지로, Scope_Definition에 없는 세부 결정(검증 코드의 입력 구성, 리포트
포맷, test-vs-validation gap 지표 도입 여부 등)이 필요해 별도 문서로 정리한다.

## 1. 목적 (Why)

Phase 3가 만든 `model_<ver>.pkl`(train set 600건으로 재학습된 최종 모델)의 실제 일반화
성능을, 그동안 어떤 단계에서도 학습·튜닝에 쓰이지 않은 `val_vectors.parquet`(200건)으로
**최초이자 유일하게** 평가한다. 이 결과가 [[Scope_Definition]] 4.4절 목표치 충족 여부와
루프백(Phase 1 또는 Phase 3 재작업) 필요성을 사람이 판단하는 근거가 된다.

## 2. 배경 및 제약

- **`val_vectors.parquet`을 이 시점까지 어떤 코드도 읽지 않았다** — [[P3_요구사항정의서_Training]]
  2절에서 Phase 3가 명시적으로 배제했던 바로 그 파일이 Phase 4의 유일한 평가 대상 데이터다.
  이 격리가 깨지면(예: 실수로 Phase 3에서 val을 참조) 평가가 낙관적으로 오염되므로, Phase 4
  구현 시에도 val set을 학습·재학습에 절대 사용하지 않는다(읽기 전용, 평가 전용).
- **Phase 3는 test set(200건)으로 하이퍼파라미터를 *선택*했다.** 즉 test set 기준 F1-macro/
  Accuracy(`hyperparams.json`의 `best_f1_macro`/`best_accuracy`)는 "그 조합이 test set에
  최적화되도록 선택된 결과"이므로 다소 낙관적 편향(optimistic bias)이 있을 수 있다. Phase 4는
  이 test 성적과 val 성적의 **차이(gap)** 를 함께 보고해, "목표 미달" 자체와는 별개로 "test에
  과적합해 선택된 결과인지"를 판단할 근거를 제공한다(사용자 확인, 2026-08-20 — gap 지표를
  요구사항에 포함하기로 결정).
- Phase 3 산출물 스키마(실측 확인, 2026-08-19): `val_vectors.parquet` 컬럼은
  `embedding`(1024D `float32` 리스트)/`label`(문자열, `CLASS_LABELS` 5종) 2개뿐 — train/test와
  동일 스키마.
- `model_<ver>.pkl`은 `training/persistence.save_model()`이 저장한 `LogisticRegressionClassifier`
  래퍼 인스턴스이며, `domain/interfaces.Classifier` Protocol(`fit`/`predict_proba`)을 만족한다.
  Phase 4는 이 Protocol에만 의존하고 sklearn을 직접 알지 않는다([[CLAUDE.md]] 1절 DIP —
  분류기를 다른 구현체로 바꿔도 Phase 4 코드는 무수정).
- `hyperparams.json`(`training/persistence.save_search_result()` 산출물, `HyperparamSearchResult`
  스키마)이 gap 지표 계산의 두 번째 입력이 된다 — Phase 4는 이 파일에서 `best_accuracy`/
  `best_f1_macro`(test set 성적)만 읽고, `trials`(전체 탐색 이력)는 사용하지 않는다.
- [[Scope_Definition]] 4.5절의 루프백(데이터 품질→Phase 1, 모델/하이퍼파라미터→Phase 3)은
  **사람이 리포트를 읽고 내리는 결정**이다. Phase 4 코드는 Phase 1/3을 자동으로 재실행하거나
  트리거하지 않는다 — 리포트에 판단 근거(목표 미달 여부, gap 크기, 오분류 패턴)를 충분히
  제공하는 것까지가 범위다.
- `evaluation/` 디렉터리는 아직 존재하지 않는다 — [[Architecture_Design]] 2절이 이미
  `evaluation/metrics.py`(accuracy/F1/confusion matrix, IT vs NON_IT 집계)와
  `evaluation/report.py`(리포트 생성)로 모듈 분리를 earmark해 두었으므로 그대로 따른다.

## 3. 범위

### In Scope

| # | 항목 |
|---|---|
| 1 | `val_vectors.parquet` 로딩 → `embedding`(1024D)/`label`(5-class) 분리 |
| 2 | `model_<ver>.pkl` 로드(`Classifier` Protocol) → val set에 대해 `predict_proba()` 실행 |
| 3 | 5-class Accuracy, IT-vs-NON_IT 이진 Accuracy, F1-macro, Confusion Matrix, 클래스별 Classification Report 산출 |
| 4 | `hyperparams.json`(Phase 3 산출물)에서 test set 성적(`best_accuracy`/`best_f1_macro`)을 읽어 val 성적과의 **gap**(test − val, Accuracy/F1-macro 각각) 계산 |
| 5 | [[Scope_Definition]] 4.4절 목표치(5-class Accuracy ≥85%, IT/NON_IT Accuracy ≥90%, F1-macro ≥0.85) 대비 달성 여부 판정(코드가 참/거짓만 표시, 루프백 실행은 하지 않음) |
| 6 | 위 지표 전체를 `eval_report_<ver>.md`(사람이 읽는 리포트) + `.json`(구조화 데이터)로 저장 |

### Out of Scope (다른 Phase 책임)

| 항목 | 담당 |
|---|---|
| 루프백 실행(데이터 재생성, 하이퍼파라미터 재탐색) — 사람이 리포트를 보고 별도로 판단·수행 | Phase 1 / Phase 3 (사람 결정) |
| 모델 학습·하이퍼파라미터 탐색 | Phase 3 (완료) |
| 실시간 추론 파이프라인(`inference/predictor.py`), Embedding Service(`localhost:8000`) 호출 | Phase 5 |
| `val_vectors.parquet` 자체 생성 | Phase 2 (완료) |

## 4. 기능 요구사항

### 4.1 입력

| 입력 | 스키마/형식 | 용도 |
|---|---|---|
| `data/<version>/val_vectors.parquet` | `embedding`(1024D `float32`), `label`(`CLASS_LABELS` 5종) | 평가 대상 데이터(200건) |
| `<model-path>.pkl` | `Classifier` Protocol 구현체(joblib 직렬화) | 평가 대상 모델 |
| `<search-result-path>.json` | `HyperparamSearchResult`(`best_accuracy`, `best_f1_macro`) | gap 지표 계산용 test set 성적 |

### 4.2 평가 지표 산출

- `model.predict_proba(X_val)` → 클래스별 확률 dict 리스트에서 최댓값 클래스를 예측 라벨로 채택.
- **5-class Accuracy**: 예측 라벨 == 실제 `label`인 비율.
- **IT vs NON_IT Accuracy**: 예측/실제 라벨을 각각 `IT_LABEL`이면 `IT`, 나머지(`NON_IT_LABELS`)면
  `NON_IT`로 집계한 뒤 정답 비율([[Scope_Definition]] 2절 최종 판정 규칙과 동일 집계 로직).
- **F1-macro**: 5-class 각각의 F1을 산출해 단순 평균.
- **Confusion Matrix**: 5-class 전체 오분류 매트릭스 + IT-vs-NON_IT 집계 매트릭스. 표현 형식
  (표/구조화 데이터, 이미지 렌더링 여부)은 설계서에서 확정한다 — 본 요구사항은 "두 레벨(5-class,
  이진)의 매트릭스 데이터가 리포트에 존재해야 한다"만 고정한다.
- **Classification Report**: 클래스별 precision/recall/f1/support.

### 4.3 Test-vs-Validation Gap 지표 (신규)

- Accuracy gap = `hyperparams.json.best_accuracy` − (본 Phase가 계산한 val 5-class Accuracy)
- F1-macro gap = `hyperparams.json.best_f1_macro` − (본 Phase가 계산한 val F1-macro)
- 두 gap 값을 그대로 리포트에 수치로 기록한다. gap이 큰 경우(과적합 의심, 즉 Phase 3
  루프백 신호) vs 작은 경우(목표 미달이라면 데이터 품질 의심, 즉 Phase 1 루프백 신호)의
  해석 기준([[project_phase4_validation_planning]] 메모 참고)은 리포트 본문에 설명 텍스트로
  포함하되, "이 정도 gap이면 경고"라는 임계값을 코드에 하드코딩하지 않는다([[CLAUDE.md]]
  4절) — 임계값을 둘지, 둔다면 설정값으로 어떻게 관리할지는 설계서에서 확정한다.

### 4.4 목표 달성 여부 판정

- [[Scope_Definition]] 4.4절 목표치 3가지(5-class Accuracy/IT-NON_IT Accuracy/F1-macro) 각각에
  대해 달성/미달 여부를 불리언으로 산출해 리포트에 포함한다.
- 미달 시 자동 재시도나 재실행은 하지 않는다 — 사람이 리포트(목표 미달 여부 + gap 크기 +
  Classification Report의 클래스별 오분류 패턴)를 보고 Phase 1/3 중 어디로 루프백할지
  판단한다([[Scope_Definition]] 4.5절).

### 4.5 리포트 생성

- `eval_report_<ver>.md`: 사람이 읽는 요약 리포트(지표 표, 목표 달성 여부, gap 수치와 해석
  가이드, Confusion Matrix, Classification Report).
- `eval_report_<ver>.json`: 위와 동일한 내용을 구조화된 형태로 저장 — 이후 자동 비교/추적에
  사용 가능해야 한다.
- 재실행 시 기존 리포트 파일을 덮어쓰지 않는다([[CLAUDE.md]] 5절 입출력 보존).

## 5. 비기능 요구사항 (품질 기준)

| 항목 | 기준 |
|---|---|
| 재현성 | 동일 `val_vectors.parquet` + 동일 `model.pkl` + 동일 `hyperparams.json` 입력 시 동일 리포트 산출(평가 로직에 비결정 요소 없음) |
| 등급/커버리지 | [[CLAUDE.md]] 2절 — `evaluation/metrics.py`는 등급 A(핵심 순수 로직, 입력 X_val/y_val/예측값 → 지표 계산만, 라인 커버리지 ≥90%), `evaluation/report.py`/`cli/run_phase4.py`는 등급 B(오케스트레이션, 라인 커버리지 ≥70%) |
| 하드코딩 금지 | 목표치([[Scope_Definition]] 4.4절 수치)와 gap 경고 임계값(도입 시)을 코드 리터럴로 두지 않음([[CLAUDE.md]] 4절) — 목표치는 설계상 고정값이므로 `constants.py` 후보, gap 임계값은 튜닝 대상이므로 설정 파일 후보(설계서에서 확정) |
| Docker 실행 | 검증 스크립트·테스트 모두 컨테이너 내부에서 실행 가능해야 함 |
| 산출물 보존 | `eval_report_<ver>.md/json`은 재실행 시 기존 파일을 덮어쓰지 않음([[CLAUDE.md]] 5절) |
| val set 격리 | `val_vectors.parquet`은 이 Phase에서만 읽고, 어떤 학습·재학습에도 사용하지 않음 |

## 6. 산출물

| 파일 | 설명 | 비고 |
|---|---|---|
| `data/<version>/val_vectors.parquet` | 입력(검증셋, 200건) | Phase 2 산출물, 재사용 |
| `<model_dir>/model_<ver>.pkl` | 입력(평가 대상 모델) | Phase 3 산출물, 재사용 |
| `<model_dir>/hyperparams_<ver>.json` | 입력(gap 계산용 test 성적) | Phase 3 산출물, 재사용 |
| `eval_report_<ver>.md` | 사람이 읽는 검증 리포트 | 신규 |
| `eval_report_<ver>.json` | 구조화된 검증 리포트 | 신규 |
| `src/embedding_lr/evaluation/{metrics,report}.py` + 대응 테스트 | 검증 파이프라인 코드+테스트 | 신규 |
| `src/embedding_lr/cli/run_phase4.py` | Phase 4 CLI 진입점 | 신규 |

## 7. 완료 기준 (Acceptance Criteria)

- [ ] `val_vectors.parquet`(200건)에 대해 5-class Accuracy/IT-NON_IT Accuracy/F1-macro/
      Confusion Matrix/Classification Report가 모두 산출됨
- [ ] `hyperparams.json`의 test 성적과 val 성적 간 gap(Accuracy, F1-macro)이 리포트에 수치로
      포함됨
- [ ] [[Scope_Definition]] 4.4절 목표치 3가지 각각의 달성/미달 여부가 리포트에 명시됨
- [ ] `eval_report_<ver>.md`와 `.json`이 동일 내용으로 생성되고, 재실행 시 기존 파일을
      덮어쓰지 않음
- [ ] Phase 4 코드가 `val_vectors.parquet`을 어떤 학습·재학습에도 사용하지 않음(읽기 전용)
      확인
- [ ] `Classifier` Protocol(`predict_proba`)에만 의존하고 sklearn을 직접 import하지 않음
      확인(단, `evaluation/metrics.py` 내부 지표 계산 함수 자체는 sklearn `metrics` 모듈
      사용 가능 — Protocol 경계는 "모델 접근"에 한함)
- [ ] Docker 컨테이너 내부에서 검증 스크립트+테스트 실행 확인
- [ ] 등급 A(`metrics.py`) 커버리지 ≥90%, 등급 B(`report.py`/`run_phase4.py`) 커버리지 ≥70%
      확인(테스트결과서에 기록)

## 8. 리스크 및 참고사항

- gap이 크게 나타나는 경우와 목표 미달이 동시에 발생하면 "Phase 3 하이퍼파라미터가 test
  set에 과적합되어 선택되었을 가능성"으로 해석하고, gap이 작은데도 목표 미달이면 "데이터
  자체(라벨 품질, 클래스 경계 모호성)" 쪽 원인일 가능성이 크다는 것이 현재까지의 해석
  가이드다([[project_phase4_validation_planning]] 메모, 2026-08-20 논의) — 다만 이 가이드는
  아직 코드 로직(예: 자동 판정 규칙)으로 확정하지 않았고, 리포트에 "참고 설명"으로만
  포함한다. 자동 판정 규칙화 여부는 설계서 단계에서 다시 논의한다.
- Confusion Matrix의 구체적 출력 형식(표 vs 이미지)은 [[Scope_Definition]] 4.4절이 "시각화
  출력"이라고만 명시해 모호하다 — 현재 프로젝트에 플로팅 라이브러리 의존성이 없으므로,
  설계서 단계에서 "구조화 데이터 + 마크다운 표"로 확정할지 이미지 렌더링을 추가할지 결정
  필요.
- 목표 미달 시 루프백은 사람이 수행하므로, Phase 4 CLI는 미달이어도 비정상 종료(exit code
  오류)하지 않는다 — 정상적으로 리포트를 생성하고 종료한다(설계서에서 종료 코드 정책 확정).
