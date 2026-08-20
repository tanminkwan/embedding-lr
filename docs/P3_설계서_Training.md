# 설계서 — Phase 3 모델 학습 (Training)

[[P3_요구사항정의서_Training]](요구사항정의서)를 [[Architecture_Design]] 2절(모듈 구조)·
4절(데이터 흐름 상세, `training.trainer` GridSearchCV)의 실제 모듈 시그니처로 구체화한
설계서. [[CLAUDE.md]] 3절 순서상 2단계 산출물이다.

대상 모듈: `training/{trainer,persistence}.py`, `cli/run_phase3.py`, `domain/models.py`
(하이퍼파라미터 탐색 결과 모델 추가).

## 1. 범위와 설계 전제

- **입력은 Phase 2 산출물 그대로**: `train_vectors.parquet`(600건)/`test_vectors.parquet`
  (200건), 컬럼은 `embedding`(1024D `list[float32]`)/`label`(문자열, `CLASS_LABELS` 5종)
  뿐이다(실측 확인, 2026-08-19). `val_vectors.parquet`은 이 Phase가 절대 읽지 않는다 —
  Phase 4 전용([[P3_요구사항정의서_Training]] 2절).
- **탐색 방식은 `GridSearchCV` + `PredefinedSplit`**([[Architecture_Design]] 4절/5절에서
  이미 확정). 다만 기본 k-fold 교차검증이 아니라 **고정 train/test 분할 자체를 유일한
  평가 폴드로 강제**한다 — [[Scope_Definition]] 4.3절이 요구하는 "테스트셋 200건 기준
  선택"과 sklearn `GridSearchCV`의 병렬 탐색·결과 테이블(`cv_results_`) 편의성을 동시에
  만족시키기 위함. `refit=False`로 두고, 조합별 성적은 `cv_results_`에서 직접 뽑아 코드가
  최종 선정한다(아래 2.3절 tie-break 규칙 때문에 `GridSearchCV`의 자동 `refit` 선택에
  맡기지 않는다).
- **최종 모델은 train set만으로 재학습**한다. `GridSearchCV`는 탐색(각 조합의 test 성적
  확인)에만 쓰고, 선정된 최적 파라미터로 새 `LogisticRegression`을 만들어 train set
  600건으로만 다시 `fit()`한다 — test set은 탐색 판단에만 쓰이고 최종 모델 학습 데이터에는
  섞이지 않는다([[P3_요구사항정의서_Training]] 4.2절 "학습 입력: train set 600건"과 일치).
- **선정 기준(tie-break 포함)**: F1-macro 내림차순 → 동률 시 Accuracy 내림차순
  ([[P3_요구사항정의서_Training]] 4.3절, 사용자 확인 2026-08-19).
- **DIP 경계**: `training/trainer.py`는 `domain/interfaces.Classifier` Protocol을 만족하는
  구현체(`LogisticRegressionClassifier`)를 제공한다. Phase 4(`evaluation/*`, 미구현)와
  Phase 5(`inference/predictor.py`, 미구현)는 이 Protocol에만 의존하고 sklearn을 직접
  모른다 — LogisticRegression을 다른 분류기로 바꿔도 두 Phase는 무수정([[CLAUDE.md]]
  1절 DIP).
- **하이퍼파라미터 탐색 범위는 코드가 아니라 JSON 설정 파일에서 읽는다**(OCP,
  [[CLAUDE.md]] 1절/4절). 저장소에 기본값 파일 `config/hyperparams_default.json`을
  두고(값 자체는 [[Scope_Definition]] 4.3절 표와 동일), CLI `--config`로 다른 경로를
  지정하면 그 파일을 대신 쓴다 — [[Architecture_Design]] 3절 Workflow 규약이 이미
  모든 Phase CLI에 `[--config <path>]`를 선택 인자로 예정해 두었다.

```
                    ┌─ Classifier (Protocol, domain/interfaces.py) ─────────┐
                    │  fit(X, y) -> None                                    │
                    │  predict_proba(X) -> list[dict[str, float]]           │
                    └───────────────────────▲────────────────────────────────┘
                                             │ 구현
                    LogisticRegressionClassifier ─┘ (sklearn LogisticRegression 래핑)

training.trainer.load_vectors(train_path) ──> (X_train, y_train)
training.trainer.load_vectors(test_path)  ──> (X_test, y_test)
        │
training.trainer.search_hyperparameters(X_train, y_train, X_test, y_test, param_grid)
        │  (GridSearchCV + PredefinedSplit, refit=False, cv_results_에서 F1-macro desc→Accuracy desc 정렬)
        ▼
HyperparamSearchResult(best_params, best_f1_macro, best_accuracy, trials=[...])
        │
training.trainer.train_final_model(X_train, y_train, best_params) ──> LogisticRegressionClassifier
        │
training.persistence.save_model(model, model_output_path)              [.pkl, joblib]
training.persistence.save_search_result(result, search_output_path)    [.json]
```

## 2. `domain/models.py` (갱신) — 하이퍼파라미터 탐색 결과 모델

```python
class HyperparamTrial(BaseModel):
    """하이퍼파라미터 조합 1개 + test set 성적 1건."""
    params: dict[str, float | int | str]
    accuracy: float
    f1_macro: float


class HyperparamSearchResult(BaseModel):
    """search_hyperparameters() 반환값. hyperparams.json으로 그대로 직렬화된다."""
    best_params: dict[str, float | int | str]
    best_accuracy: float
    best_f1_macro: float
    trials: list[HyperparamTrial]
```

- `trials`는 탐색한 **모든** 조합을 담는다(선정된 조합 포함) — 이후 재현·비교·감사가
  가능해야 한다는 요구사항([[P3_요구사항정의서_Training]] 4.5절)을 만족한다. 아무 조합도
  가만히 버리지 않는다(누락 없는 완전한 이력).
- 등급 A(순수 데이터 모델, [[CLAUDE.md]] 2절 `domain/models` 분류와 동일) — 필드 검증만
  있고 로직은 없다.

## 3. `domain/interfaces.py` (갱신 없음, 계약 재확인)

[[P0_설계서_Common]] 4절에 이미 정의된 `Classifier` Protocol을 그대로 쓴다.

```python
class Classifier(Protocol):
    def fit(self, X: list[list[float]], y: list[str]) -> None: ...
    def predict_proba(self, X: list[list[float]]) -> list[dict[str, float]]: ...
```

## 4. `training/trainer.py` (등급 B — [[CLAUDE.md]] 2절 표에 이미 명시된 분류)

```python
class LogisticRegressionClassifier:
    """Classifier Protocol 구현체 — sklearn LogisticRegression 래핑."""

    def __init__(self, **params) -> None:
        """params는 GridSearchCV 탐색 대상(C/solver/max_iter)만 받는다.
        multi_class="multinomial"은 이 클래스가 고정 지정(Scope_Definition 4.1절)."""

    def fit(self, X: list[list[float]], y: list[str]) -> None: ...

    def predict_proba(self, X: list[list[float]]) -> list[dict[str, float]]:
        """sklearn predict_proba()의 ndarray를 model.classes_ 순서에 맞춰
        라벨명 → 확률 dict 리스트로 변환."""


def load_vectors(path: str) -> tuple[list[list[float]], list[str]]:
    """<split>_vectors.parquet을 읽어 (embedding 컬럼, label 컬럼) 반환.
    컬럼 누락 또는 label 값이 CLASS_LABELS에 없으면 DataValidationError."""


def search_hyperparameters(
    X_train: list[list[float]], y_train: list[str],
    X_test: list[list[float]], y_test: list[str],
    param_grid: dict[str, list],
) -> HyperparamSearchResult:
    """GridSearchCV(estimator=LogisticRegression(multi_class="multinomial"),
    param_grid=param_grid, scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
    cv=PredefinedSplit(...), refit=False)로 X_train+X_test를 합쳐 fit() — PredefinedSplit이
    train 인덱스는 -1(학습 전용), test 인덱스는 0(평가 전용)으로 지정되어 있어 실제로는
    "train으로 학습, test로 평가"만 수행되고 k-fold 교차검증은 일어나지 않는다.
    cv_results_에서 조합별 (params, mean_test_f1_macro, mean_test_accuracy)를 꺼내
    HyperparamTrial 리스트로 변환한 뒤, F1-macro 내림차순 → Accuracy 내림차순으로 정렬해
    1위 조합을 best로 선정."""


def train_final_model(
    X_train: list[list[float]], y_train: list[str], best_params: dict[str, float | int | str],
) -> LogisticRegressionClassifier:
    """best_params로 새 LogisticRegressionClassifier를 만들어 train set(X_train, y_train)만
    으로 재학습 — search_hyperparameters()가 GridSearchCV 내부에서 만든 추정기를 재사용하지
    않는다(재사용 시 train+test 결합 데이터로 학습된 것이라 4.2절 요구사항과 어긋남)."""
```

- `search_hyperparameters`가 `refit=False`인 이유: sklearn의 다중 스코어러 자동 `refit`은
  스코어러 하나(`refit="f1_macro"`)만 최댓값 기준으로 고르고, F1-macro 동률일 때 Accuracy로
  가르는 우리 tie-break 규칙을 표현할 수 없다. 그래서 `cv_results_` 원본 표를 코드가 직접
  정렬해 최적 조합을 고른다.
- `PredefinedSplit` 구성: `test_fold = [-1]*len(X_train) + [0]*len(X_test)`,
  `GridSearchCV(..., cv=PredefinedSplit(test_fold))`, `X = X_train + X_test`,
  `y = y_train + y_test`로 결합해 전달. 폴드가 정확히 1개뿐이므로 `mean_test_*`는 사실상
  "그 조합의 test set 단일 평가값"과 같다.

## 5. `training/persistence.py` (등급 B)

```python
def save_model(model: LogisticRegressionClassifier, path: str) -> None:
    """joblib.dump(model, path). path가 이미 존재하면 DataValidationError
    (덮어쓰기 금지 — CLAUDE.md 5절 입출력 보존, embedding/pipeline.py와 동일 패턴)."""


def load_model(path: str) -> LogisticRegressionClassifier:
    """joblib.load(path). 파일이 없으면 ModelNotFoundError(이미 exceptions.py에 정의됨,
    Phase 5 추론 서비스 기동 시 재사용)."""


def save_search_result(result: HyperparamSearchResult, path: str) -> None:
    """result.model_dump_json(indent=2)를 path에 저장. path가 이미 존재하면
    DataValidationError."""
```

- `save_model`은 `LogisticRegressionClassifier` 래퍼 인스턴스를 그대로 직렬화한다(sklearn
  `LogisticRegression` 객체를 벗겨서 저장하지 않음) — `load_model()`이 바로 `Classifier`
  Protocol을 만족하는 객체를 반환해야 Phase 5(`inference/predictor.py`, 미구현)가 어댑팅
  없이 즉시 쓸 수 있다.
- 새 예외 타입을 만들지 않는다 — 파일 존재 충돌은 `DataValidationError`(이미
  `embedding/pipeline.py`가 출력 경로 존재 검사에 쓰는 것과 동일 패턴), 모델 파일 부재는
  기존 `ModelNotFoundError`를 재사용한다.

## 6. `config/hyperparams_default.json` (신규 설정 파일, 코드 아님)

```json
{
  "C": [0.01, 0.1, 1.0, 10.0],
  "solver": ["lbfgs", "liblinear"],
  "max_iter": [500, 1000, 2000]
}
```

- 값은 [[Scope_Definition]] 4.3절 표와 동일 — 다만 이 값들은 Python 코드가 아니라 이
  JSON 파일에만 존재한다. `cli/run_phase3.py`가 `--config` 인자(기본값이 이 파일 경로)로
  읽어 `search_hyperparameters()`에 그대로 넘긴다. 탐색 범위를 바꾸고 싶으면 이 파일(또는
  `--config`로 지정한 다른 파일)만 고치면 되고 `trainer.py`는 수정하지 않는다([[CLAUDE.md]]
  1절 OCP).

## 7. `cli/run_phase3.py`

```
Trigger: python -m embedding_lr.cli.run_phase3 \
           --train <path> --test <path> \
           --model-output <path> --search-output <path> \
           [--config <path, 기본값 config/hyperparams_default.json>]
Input:   --train(train_vectors.parquet), --test(test_vectors.parquet), --config(선택)
Output:  --model-output(.pkl), --search-output(.json) — 둘 다 이미 존재하면 실패
```

- `run_phase1(_5)/run_phase2.py`와 동일한 컨벤션: `Settings()` → `setup_logging()` →
  `run_context("phase3", settings)`로 상태 파일(`status/phase3_<run_id>.json`) 기록.
- 순서: `trainer.load_vectors(train)` → `trainer.load_vectors(test)` → `--config` JSON 로드
  → `trainer.search_hyperparameters(...)` → `trainer.train_final_model(...)` →
  `persistence.save_model(...)` + `persistence.save_search_result(...)`.
- Phase 2와 달리 외부 서비스(AIPro+ 등) 호출이 없으므로 `finally`의 커넥션 정리 로직은
  불필요하다.

## 8. 데이터 흐름 요약

```
data/<version>/train_vectors.parquet ──┐
data/<version>/test_vectors.parquet ───┼─(trainer.load_vectors)──> (X_train,y_train)/(X_test,y_test)
config/hyperparams_default.json (또는 --config) ─┘
        │
   trainer.search_hyperparameters (GridSearchCV + PredefinedSplit, refit=False)
        │  cv_results_ → F1-macro desc, Accuracy desc 정렬 → best_params
        ▼
   trainer.train_final_model (train set만 재학습) ──> LogisticRegressionClassifier
        │
        ├─(persistence.save_model)──> <model-output>.pkl
        └─(persistence.save_search_result)──> <search-output>.json (HyperparamSearchResult)
                                                          [cli/run_phase3.py]
```

## 9. 테스트 등급 및 완료 기준

| 모듈 | 등급([[CLAUDE.md]] 2절) | 목표 커버리지 | 비고 |
|---|---|---|---|
| `domain/models.py` (`HyperparamTrial`/`HyperparamSearchResult` 추가분) | A | ≥ 90% | 순수 데이터 모델 — 필드 검증 케이스 |
| `training/trainer.py` | B | ≥ 70% | fixture parquet(`tmp_path`)로 통합 테스트 — `load_vectors` 스키마 오류, tie-break(F1 동률 시 Accuracy) 분기 포함 |
| `training/persistence.py` | B | ≥ 70% | `tmp_path`로 save/load 왕복 + 기존 파일 존재 시 실패 + `ModelNotFoundError` 케이스 |
| `cli/run_phase3.py` | B | ≥ 70% | fixture parquet + fixture config JSON으로 end-to-end 실행, 실제 AIPro+/외부 서비스 호출 없음(애초에 이 Phase는 호출하지 않음) |

완료 기준: 위 4개 대상이 각 목표 커버리지를 충족하고, (1) val_vectors.parquet을 어떤
경로에서도 읽지 않음, (2) 최종 모델이 train set만으로 재학습됨, (3) F1-macro 우선
tie-break가 Accuracy로 넘어가는 케이스가 테스트로 확인되면 Phase 3 코드는 완료로
간주한다. 실측치는 이후 작성할 `P3_테스트결과서_Training.md`에 기록한다.

## 10. 관련 문서/코드

- 요구사항: [[P3_요구사항정의서_Training]]
- 상위 설계: [[Architecture_Design]] 2절(모듈 구조), 4절(데이터 흐름, GridSearchCV 확정),
  5절(`training/persistence.py`=joblib, mlflow는 도입 보류)
- 공통 모듈: [[P0_설계서_Common]] 4절(`Classifier` Protocol), 5절(`ModelNotFoundError`/
  `DataValidationError`)
- 관련 코드: `src/embedding_lr/training/{trainer,persistence}.py`,
  `src/embedding_lr/cli/run_phase3.py`, `src/embedding_lr/domain/models.py`(갱신),
  `config/hyperparams_default.json`(신규)
- 테스트: `tests/unit/test_models.py`(갱신), `tests/integration/test_{trainer,persistence,
  run_phase3}.py`
- 신규 의존성: `scikit-learn`, `joblib` — `pyproject.toml`에 고정 버전으로 추가 필요
  ([[P3_요구사항정의서_Training]] 2절 리스크).
