# P3_요구사항정의서_Training — Phase 3 요구사항 정의서

[[Scope_Definition]] 4절(분류 모델 설계)과 7절(Phase 로드맵)을 Phase 3 단위로 구체화한
요구사항 정의서. [[CLAUDE.md]] 3절 "작업 진행 순서"의 1단계 산출물이며, 다음 산출물은
`P3_설계서_Training.md`(설계서)에서 다룬다.

Phase 0/2([[P0_설계서_Common]], [[P2_설계서_Embedding]])는 Scope_Definition·Architecture_Design이
이미 요구사항을 상세히 정의해 별도 요구사항정의서 없이 설계서로 직행했으나, Phase 3는
Phase 1과 마찬가지로 Scope_Definition에 없는 세부 결정(탐색 범위의 설정화 방식, 산출물
경로/버전 규칙, Phase 3/4 책임 경계 등)이 필요해 별도 문서로 정리한다.

## 1. 목적 (Why)

Phase 2가 만든 `train_vectors.parquet`/`test_vectors.parquet`(1024D 임베딩 + 5-class 라벨)을
입력으로 Logistic Regression 다중분류 모델을 학습하고, 하이퍼파라미터 탐색을 통해 test set
기준 최적 조합을 선정하여 `.pkl` 모델 파일로 저장한다. 이 모델이 Phase 4(검증)와 Phase
5(추론)의 유일한 입력이므로, 여기서 확정하는 학습·탐색·저장 절차가 이후 Phase의 재현성과
신뢰도를 좌우한다.

## 2. 배경 및 제약

- Phase 2가 이미 `data/<version>/{train,test,val}_vectors.parquet`을 생성 완료했다. 실제
  스키마를 확인한 결과(2026-08-19, Docker 컨테이너 내부에서 pandas로 조회) 컬럼은
  `embedding`(1024차원 `float32` 리스트)과 `label`(문자열, `CLASS_LABELS` 5종 중 하나)
  2개뿐이다. `source`가 아니라 `label`이라는 컬럼명으로 저장되어 있음에 주의.
- **`val_vectors.parquet`은 존재하지만 Phase 3에서는 사용하지 않는다.** [[Scope_Definition]]
  7절 로드맵상 최종 성능 평가(Accuracy/F1, 목표 미달 시 루프백)는 **Phase 4** 책임이고,
  Phase 3는 **test set(200건)만으로 하이퍼파라미터 조합을 선정**한다([[Scope_Definition]]
  4.3절: "테스트셋 200건에 대한 정확도(Accuracy)와 F1-Score 기준으로 최적 조합 선택"). 이
  경계를 문서로 명확히 해 두지 않으면 구현 시 validation set이 조기에 소진될 위험이 있다.
- `domain/interfaces.py`에 Phase 0에서 이미 `Classifier` Protocol(`fit`/`predict_proba`)이
  선언되어 있다. Phase 3 구현체는 이 Protocol을 만족해야 [[CLAUDE.md]] 1절(DIP) 원칙이
  유지된다 — 특히 `predict_proba`는 sklearn의 raw `ndarray`가 아니라 `list[dict[str, float]]`
  (라벨명 → 확률) 형태로 반환해야 한다.
- [[CLAUDE.md]] 1절(OCP)에 따라 하이퍼파라미터 탐색 범위(`C`/`solver`/`max_iter`)는 코드
  수정 없이 설정으로 확장 가능해야 한다. [[Scope_Definition]] 4.3절의 표는 탐색 범위의
  "예시 값"이지, 코드 리터럴로 고정하라는 뜻이 아니다.
- [[CLAUDE.md]] 5절(Workflow 친화적 구성, 입출력 보존)에 따라 학습 산출물(모델 파일,
  탐색 이력)은 재실행 시 기존 파일을 덮어쓰지 않아야 한다.
- `pyproject.toml`에 아직 `scikit-learn`/`joblib` 의존성이 없다 — 설계서 단계에서 추가
  필요(고정 버전 관리, [[CLAUDE.md]] 6절).

## 3. 범위

### In Scope

| # | 항목 |
|---|---|
| 1 | `train_vectors.parquet` 로딩 → `embedding`(1024D)/`label`(5-class) 분리 |
| 2 | Logistic Regression(`multi_class='multinomial'`) 학습 — `Classifier` Protocol 구현체 |
| 3 | 하이퍼파라미터 조합별 학습 → `test_vectors.parquet`로 Accuracy/F1(macro) 평가 → 최적 조합 선정 |
| 4 | 탐색 범위(`C`/`solver`/`max_iter`)를 코드 수정 없이 설정으로 조정 가능하게 구성 |
| 5 | 선정된 최적 모델을 `.pkl`(joblib)로 저장, 재실행 시 기존 파일 미덮어쓰기 |
| 6 | 하이퍼파라미터 탐색 이력(조합별 파라미터 + 평가지표) 구조화 기록 저장 |

### Out of Scope (다른 Phase 책임)

| 항목 | 담당 |
|---|---|
| `val_vectors.parquet` 기반 최종 성능 평가, Confusion Matrix, Classification Report | Phase 4 |
| 목표 미달 시 루프백 판단·실행 | Phase 4 |
| 추론 파이프라인(`inference/predictor.py`), Embedding Service(`localhost:8000`) 호출 | Phase 5 |
| 임베딩 생성·저장(`*_vectors.parquet` 자체) | Phase 2 (완료) |

## 4. 기능 요구사항

### 4.1 입력 스키마 (`data/<version>/{train,test}_vectors.parquet`)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `embedding` | `list[float32]`, 길이 1024 | BGE-M3 임베딩 벡터 |
| `label` | string (enum) | `CLASS_LABELS`(`IT`/`DAILY`/`KNOWLEDGE`/`CREATIVE`/`ANOMALY`) 중 하나 |

### 4.2 모델 학습

- scikit-learn `LogisticRegression(multi_class='multinomial')`로 5-class 동시 학습(One-vs-Rest 아님).
- 학습 입력: train set 600건(클래스당 120건).
- `Classifier` Protocol 계약 준수: `fit(X: list[list[float]], y: list[str])`,
  `predict_proba(X) -> list[dict[str, float]]`(라벨명 → 확률 dict로 어댑팅).

### 4.3 하이퍼파라미터 탐색

- 탐색 대상: `C`, `solver`, `max_iter` (기본 범위는 [[Scope_Definition]] 4.3절 표 참고,
  코드가 아닌 설정에서 관리).
- 선정 기준: test set(200건) Accuracy + F1-macro.
- 탐색 방식(`GridSearchCV` vs 수동 반복 비교)은 [[Scope_Definition]]에서도 선택지로 열려
  있으므로 설계서에서 확정한다 — 본 요구사항은 "탐색 범위 설정 가능 + test set 기준 선정"만
  고정한다.

### 4.4 모델 저장

- 선정된 최적 모델을 joblib으로 직렬화해 `.pkl`로 저장.
- 재실행 시 기존 산출물을 덮어쓰지 않는다 — 버전/타임스탬프 등 구분 규칙은 설계서에서 확정.

### 4.5 탐색 이력 기록

- 각 하이퍼파라미터 조합 + test set Accuracy/F1을 구조화된 형태(예: JSON/CSV)로 남겨,
  이후 재현·비교·감사가 가능하게 한다.

## 5. 비기능 요구사항 (품질 기준)

| 항목 | 기준 |
|---|---|
| 재현성 | 동일 입력(`train`/`test_vectors.parquet`) + 동일 설정으로 재학습 시 동일 최적 조합이 선정됨(`random_state` 등 고정) |
| 등급/커버리지 | [[CLAUDE.md]] 2절 등급 B(오케스트레이션) 적용 — 구현 후 통합 테스트, 라인 커버리지 ≥ 70% |
| 하드코딩 금지 | 탐색 범위·기본 하이퍼파라미터 값을 코드 리터럴로 두지 않음([[CLAUDE.md]] 4절) |
| Docker 실행 | 학습 스크립트·테스트 모두 `docker/Dockerfile.pipeline` 컨테이너 내부에서 실행 가능해야 함 |
| 산출물 보존 | 모델 파일·탐색 이력 파일은 재실행 시 기존 파일을 덮어쓰지 않음([[CLAUDE.md]] 5절) |

## 6. 산출물

| 파일 | 설명 | 비고 |
|---|---|---|
| `data/<version>/train_vectors.parquet` | 입력(학습셋, 600건) | Phase 2 산출물, 재사용 |
| `data/<version>/test_vectors.parquet` | 입력(탐색용 평가셋, 200건) | Phase 2 산출물, 재사용 |
| `<model_dir>/...` `.pkl` | 학습된 최적 LR 모델 | 신규 — 경로/버전 규칙은 설계서에서 확정 |
| 하이퍼파라미터 탐색 이력 파일 | 조합별 파라미터 + Accuracy/F1 | 신규 — 포맷/경로는 설계서에서 확정 |
| `src/embedding_lr/training/*.py` + 대응 테스트 | 학습 파이프라인 코드+테스트 | 신규 |

## 7. 완료 기준 (Acceptance Criteria)

- [ ] `train_vectors.parquet`(600건)으로 학습, `test_vectors.parquet`(200건)으로 하이퍼파라미터별 Accuracy/F1 산출
- [ ] 탐색 범위(`C`/`solver`/`max_iter`)가 코드 수정 없이 설정 변경만으로 조정 가능함을 확인
- [ ] 최적 조합으로 학습한 모델이 `.pkl`로 저장되고 joblib으로 재로드 가능함을 확인
- [ ] 탐색 이력 파일에 모든 시도 조합 + 지표가 기록됨을 확인
- [ ] `Classifier` Protocol 계약(`fit`/`predict_proba`) 준수 확인(`predict_proba`가 라벨명 dict 반환)
- [ ] Docker 컨테이너 내부에서 학습 스크립트+테스트 실행 확인
- [ ] 등급 B 커버리지 ≥ 70% 확인(테스트결과서에 기록)

## 8. 리스크 및 참고사항

- `val_vectors.parquet`을 Phase 3 구현 중 실수로 사용하지 않도록 주의 — Phase 4 책임과
  명확히 분리한다(2절 참고).
- [[Scope_Definition]] 4.4절의 목표 성능(5-class Accuracy ≥ 85%, IT/NON_IT Accuracy ≥ 90%,
  F1-macro ≥ 0.85)은 **Phase 4 최종 검증(validation set) 기준**이다. Phase 3의 목표는 test
  set에서 상대적으로 더 나은 하이퍼파라미터 조합을 고르는 것이며, 이 조합으로도 Phase 4에서
  목표 미달 시 [[Scope_Definition]] 4.5절에 따라 Phase 3(파라미터 재조정) 또는 Phase
  1(데이터 품질)로 루프백한다.
- `scikit-learn`/`joblib`이 `pyproject.toml`에 아직 없으므로 설계서 단계에서 버전을 고정해
  추가해야 한다.
