# 설계서 — Phase 4 검증 (Validation)

[[P4_요구사항정의서_Validation]](요구사항정의서)를 [[Architecture_Design]] 2절(모듈 구조,
`evaluation/{metrics,report}.py` earmark)의 실제 모듈 시그니처로 구체화한 설계서.
[[CLAUDE.md]] 3절 순서상 2단계 산출물이다.

대상 모듈(신규): `evaluation/{metrics,report}.py`, `cli/run_phase4.py`,
`config/eval_thresholds_default.json`. 대상 모듈(갱신): `domain/models.py`,
`constants.py`, `training/persistence.py`(`load_search_result` 추가).

사용자 확인(2026-08-20, 세 가지 설계 결정):
1. gap 경고는 **설정 가능한 임계값**으로 플래그를 자동 표시한다(하드코딩 금지, [[CLAUDE.md]] 4절).
2. Confusion Matrix는 **구조화 데이터 + 마크다운 표**로만 담는다(이미지 렌더링 없음, 신규
   플로팅 의존성 추가하지 않음).
3. 목표 미달이어도 `cli/run_phase4.py`는 **항상 종료 코드 0**으로 끝난다(루프백은 사람이
   리포트를 보고 판단 — [[Scope_Definition]] 4.5절).

## 1. 범위와 설계 전제

- **입력은 세 파일**: `val_vectors.parquet`(Phase 2), `model_<ver>.pkl`(Phase 3),
  `hyperparams_<ver>.json`(Phase 3, `HyperparamSearchResult`). Phase 4는 이 중 어느 것도
  다시 계산하거나 재학습하지 않는다 — 순수 평가만 수행한다.
- **모델 접근은 `Classifier` Protocol을 통해서만** — `evaluation/report.py`는
  `training.persistence.load_model()`이 반환한 객체의 `predict_proba()`만 호출한다.
  sklearn을 직접 import하지 않는다([[CLAUDE.md]] 1절 DIP, [[P4_요구사항정의서_Validation]]
  7절 완료 기준과 동일 — 단, `evaluation/metrics.py` 내부 지표 계산 자체는 sklearn
  `metrics` 서브모듈을 사용해도 무방하다. Protocol 경계는 "모델 인스턴스 접근"에만
  적용된다).
- **목표치(target)는 `constants.py`에 고정값으로, gap 경고 임계값은 JSON 설정 파일로** —
  둘 다 숫자지만 성격이 다르다([[CLAUDE.md]] 4절): 5-class Accuracy ≥85%/IT-NON_IT
  Accuracy ≥90%/F1-macro ≥0.85는 [[Scope_Definition]] 4.4절이 **설계상 확정**한 값이라
  도메인 상수([[P3_요구사항정의서_Training]]의 `CLASS_LABELS`와 같은 성격)로 `constants.py`에
  둔다. 반면 gap 경고 임계값(몇 %p 이상을 "과적합 의심"으로 볼지)은 이 프로젝트가 처음
  도입하는 **튜닝 대상**이므로, `training/trainer.py`의 하이퍼파라미터 탐색 범위와 같은
  패턴으로 `config/eval_thresholds_default.json`(JSON, `--config`로 교체 가능)에 둔다.
- **재사용, 새 로더 최소 추가**: `training.trainer.load_vectors()`(이미 범용 시그니처,
  `<split>_vectors.parquet` → `(X, y)`)와 `training.persistence.load_model()`을 그대로
  재사용한다. `hyperparams.json`을 읽는 로더만 `training/persistence.py`에
  `load_search_result()`로 신규 추가한다(`save_search_result()`와 대칭, 같은 파일에 두는
  것이 SRP상 자연스럽다 — "탐색 이력 파일의 저장/로드"라는 하나의 책임).

```
                    ┌─ Classifier (Protocol, domain/interfaces.py) ─────────┐
                    │  predict_proba(X) -> list[dict[str, float]]           │
                    └───────────────────────▲────────────────────────────────┘
                                             │ 구현(재사용, 갱신 없음)
                              LogisticRegressionClassifier

training.trainer.load_vectors(val_path)            ──> (X_val, y_val)
training.persistence.load_model(model_path)        ──> model: Classifier
training.persistence.load_search_result(search_path) ──> HyperparamSearchResult
        │
evaluation.metrics.probs_to_labels(model.predict_proba(X_val)) ──> y_pred
evaluation.metrics.compute_metrics(y_val, y_pred)               ──> ValidationMetrics
evaluation.metrics.compute_gap(search_result, ValidationMetrics)──> GapMetrics
evaluation.metrics.check_targets(ValidationMetrics)             ──> TargetCheckResult
        │
evaluation.report.build_report(...) ──> EvaluationReport
evaluation.report.save_report(report, md_path, json_path) ──> eval_report_<ver>.md / .json
                                                          [cli/run_phase4.py]
```

## 2. `constants.py` (갱신) — 목표치 상수

```python
TARGET_ACCURACY = 0.85          # 5-class Accuracy 목표 (Scope_Definition 4.4절)
TARGET_BINARY_ACCURACY = 0.90   # IT vs NON_IT Accuracy 목표
TARGET_F1_MACRO = 0.85          # F1-macro 목표
```

- 세 값 모두 [[Scope_Definition]] 4.4절 표의 수치를 그대로 옮긴 것 — 설계상 고정값이므로
  환경변수(`.env`)가 아니라 상수 모듈에 둔다([[CLAUDE.md]] 4절).

## 3. `domain/models.py` (갱신) — 검증 결과 모델

```python
class ValidationMetrics(BaseModel):
    """evaluation.metrics.compute_metrics() 반환값 — val set 기준 5-class + 이진 지표."""
    accuracy: float
    f1_macro: float
    binary_accuracy: float
    confusion_matrix_labels: list[str]              # CLASS_LABELS 순서 고정
    confusion_matrix: list[list[int]]                # 5x5
    binary_confusion_matrix: list[list[int]]         # 2x2, 순서=["IT", "NON_IT"]
    classification_report: dict[str, dict[str, float]]  # 라벨 -> {precision, recall, f1, support}


class GapMetrics(BaseModel):
    """evaluation.metrics.compute_gap() 반환값 — Phase 3 test 성적 대비 val 성적 차이."""
    accuracy_gap: float       # hyperparams.best_accuracy - ValidationMetrics.accuracy
    f1_macro_gap: float       # hyperparams.best_f1_macro - ValidationMetrics.f1_macro
    warning: bool             # 두 gap 중 하나라도 임계값 초과 시 True


class TargetCheckResult(BaseModel):
    """evaluation.metrics.check_targets() 반환값 — Scope_Definition 4.4절 목표 달성 여부."""
    accuracy_target_met: bool
    binary_accuracy_target_met: bool
    f1_macro_target_met: bool


class EvaluationReport(BaseModel):
    """evaluation.report.build_report() 반환값. eval_report_<ver>.json으로 그대로 직렬화된다."""
    metrics: ValidationMetrics
    gap: GapMetrics
    targets: TargetCheckResult
```

- 등급 A(순수 데이터 모델, [[CLAUDE.md]] 2절 `domain/models` 분류와 동일) — 필드 검증만
  있고 로직은 없다. 버전 문자열은 이 모델에 담지 않는다 — Phase 3의 `model_<ver>.pkl`/
  `hyperparams_<ver>.json`과 동일하게, 버전은 출력 파일명(`eval_report_<ver>.md/json`)
  으로만 구분하고 파일 내용에는 중복 기재하지 않는다.

## 4. `evaluation/metrics.py` (신규, 등급 A — [[CLAUDE.md]] 2절 표에 이미 명시된 분류)

Trigger: 아래 함수들을 `evaluation/report.py`가 순서대로 호출. 외부 의존성 없이
`(y_true, y_pred)` 또는 이미 계산된 값만 입력받는 **순수 함수**로 구성 — 모델 로드나
파일 I/O는 하지 않는다.

```python
def probs_to_labels(probs: list[dict[str, float]]) -> list[str]:
    """각 레코드의 확률 dict에서 최댓값 키(라벨)를 선택. 동률 시 CLASS_LABELS 순서상
    먼저 나오는 라벨을 선택(결정적 tie-break, 재현성 보장)."""


def to_binary_labels(labels: list[str]) -> list[str]:
    """IT_LABEL이면 "IT", 그 외(NON_IT_LABELS)면 "NON_IT"로 매핑."""


def compute_metrics(y_true: list[str], y_pred: list[str]) -> ValidationMetrics:
    """sklearn.metrics.accuracy_score/f1_score(average="macro")/confusion_matrix(
    labels=CLASS_LABELS)/classification_report(output_dict=True)로 5-class 지표를 계산하고,
    to_binary_labels()를 양쪽에 적용해 이진 Accuracy/Confusion Matrix를 추가 계산한 뒤
    ValidationMetrics로 반환."""


def compute_gap(search_result: HyperparamSearchResult, val_metrics: ValidationMetrics,
                 gap_warning_threshold: float) -> GapMetrics:
    """accuracy_gap = search_result.best_accuracy - val_metrics.accuracy,
    f1_macro_gap = search_result.best_f1_macro - val_metrics.f1_macro.
    두 gap 중 하나라도 gap_warning_threshold를 초과하면 warning=True."""


def check_targets(val_metrics: ValidationMetrics) -> TargetCheckResult:
    """val_metrics.accuracy/binary_accuracy/f1_macro를 constants.py의
    TARGET_ACCURACY/TARGET_BINARY_ACCURACY/TARGET_F1_MACRO와 비교(>=)."""
```

- `probs_to_labels`의 tie-break(동률 시 `CLASS_LABELS` 순서)는 [[P3_설계서_Training]]의
  F1-macro→Accuracy tie-break와 같은 이유로 존재 — 같은 입력이면 항상 같은 출력이어야
  한다는 등급 A 재현성 요건([[CLAUDE.md]] 2절 표) 때문에, "동률 시 임의 선택"을 허용하지
  않는다.
- `gap_warning_threshold`는 함수 인자로 주입한다(모듈 내부 상수로 두지 않음) — 설정 파일
  값을 그대로 전달만 받으므로 `metrics.py`는 설정 파일의 존재 자체를 몰라도 된다(SRP).

## 5. `evaluation/report.py` (신규, 등급 B — [[CLAUDE.md]] 2절 표에 이미 명시된 분류)

```python
def build_report(
    model: Classifier, X_val: list[list[float]], y_val: list[str],
    search_result: HyperparamSearchResult, gap_warning_threshold: float,
) -> EvaluationReport:
    """model.predict_proba(X_val) -> metrics.probs_to_labels() -> metrics.compute_metrics()
    -> metrics.compute_gap()/check_targets()를 순서대로 호출해 EvaluationReport 조립."""


def render_markdown(report: EvaluationReport) -> str:
    """EvaluationReport를 사람이 읽는 md로 렌더링 — 지표 표, 목표 달성 여부, gap 수치 +
    warning 플래그, Confusion Matrix(5-class/이진 각각 마크다운 표), Classification
    Report(클래스별 표). 이미지는 생성하지 않는다(사용자 확인 2026-08-20)."""


def save_report(report: EvaluationReport, md_path: str, json_path: str) -> None:
    """render_markdown(report)를 md_path에, report.model_dump_json(indent=2)를 json_path에
    저장. 둘 중 하나라도 이미 존재하면 DataValidationError(둘 다 쓰기 전에 먼저 존재 여부를
    확인 — 부분 쓰기로 md만 생성되고 json은 실패하는 상황 방지)."""
```

- `render_markdown`의 gap 설명 텍스트는 [[P4_요구사항정의서_Validation]] 8절의 해석
  가이드(gap 크고 목표 미달 → Phase 3 의심, gap 작고 목표 미달 → Phase 1 의심)를 고정
  문구로 포함한다 — 이 문구는 조건 분기 없이 항상 동일하게 노출되는 "참고 설명"이며,
  `warning` 플래그만 gap 값에 따라 조건부로 강조 표시(예: `⚠`)한다.
- `save_report`가 두 경로를 모두 사전 확인하는 이유: `training/persistence.py`의 개별
  파일 존재 검사 패턴과 동일하되, 리포트는 md/json 두 파일이 항상 쌍으로 존재해야
  하므로(요구사항 4.5절) 하나만 먼저 쓰고 둘째에서 실패하는 반쪽짜리 산출물을 만들지
  않는다.

## 6. `training/persistence.py` (갱신) — 탐색 이력 로더 추가

```python
def load_search_result(path: str) -> HyperparamSearchResult:
    """path의 JSON을 읽어 HyperparamSearchResult로 파싱. 파일이 없으면 ModelNotFoundError
    (model.pkl 부재와 같은 성격의 오류 — 재사용, 신규 예외 타입 추가하지 않음)."""
```

- `save_search_result()`와 대칭 — 같은 파일(`hyperparams.json`)의 저장/로드를 한 모듈이
  전담한다. Phase 4가 이 함수를 재사용하므로 `evaluation/` 쪽에 별도 파싱 로직을 두지
  않는다(DRY).

## 7. `config/eval_thresholds_default.json` (신규 설정 파일, 코드 아님)

```json
{
  "gap_warning_threshold": 0.1
}
```

- `accuracy_gap`/`f1_macro_gap` 둘 다에 동일 임계값을 적용한다(현재로선 지표별로 다른
  임계값을 요구할 근거가 없음 — 필요해지면 이 파일에 키만 추가하면 되고 코드 수정은
  불필요, OCP).
- `cli/run_phase4.py`가 `--config`(기본값이 이 파일 경로)로 읽어 `metrics.compute_gap()`에
  그대로 전달한다. [[P3_설계서_Training]] 6절의 `hyperparams_default.json` 패턴과 동일.

## 8. `cli/run_phase4.py`

```
Trigger: python -m embedding_lr.cli.run_phase4 \
           --val <path> --model <path> --search-result <path> \
           --report-md <path> --report-json <path> \
           [--config <path, 기본값 config/eval_thresholds_default.json>]
Input:   --val(val_vectors.parquet), --model(model_<ver>.pkl),
         --search-result(hyperparams_<ver>.json), --config(선택)
Output:  --report-md(.md), --report-json(.json) — 둘 다 이미 존재하면 실패
Exit code: 항상 0 — 목표 미달 여부는 EvaluationReport.targets 필드로만 표시되고
           프로세스 종료 코드에는 반영하지 않는다(사용자 확인 2026-08-20).
```

- `run_phase1(_5)/run_phase2/run_phase3.py`와 동일한 컨벤션: `Settings()` →
  `setup_logging()` → `run_context("phase4", settings)`로 상태 파일
  (`status/phase4_<run_id>.json`) 기록.
- 순서: `trainer.load_vectors(val)` → `persistence.load_model(model)` →
  `persistence.load_search_result(search_result)` → `--config` JSON 로드 →
  `report.build_report(...)` → `report.save_report(...)`.
- Phase 3와 마찬가지로 외부 서비스(AIPro+ 등) 호출이 없다.
- 로그에 `targets`(목표 달성 여부)와 `gap.warning`을 `extra`로 남겨, 종료 코드에 반영되지
  않는 정보도 운영 로그로는 추적 가능하게 한다.

## 9. 데이터 흐름 요약

```
data/<version>/val_vectors.parquet ──┐
<model_dir>/model_<ver>.pkl ─────────┼─(trainer.load_vectors / persistence.load_model)
<model_dir>/hyperparams_<ver>.json ──┘        │
config/eval_thresholds_default.json (또는 --config) ─┐
                                                       │
                            evaluation.report.build_report
                                   │  model.predict_proba → metrics.probs_to_labels
                                   │  → metrics.compute_metrics (5-class + 이진)
                                   │  → metrics.compute_gap (test vs val)
                                   │  → metrics.check_targets (Scope_Definition 4.4절)
                                   ▼
                            EvaluationReport
                                   │
                            evaluation.report.save_report
                                   ├──> eval_report_<ver>.md  (render_markdown)
                                   └──> eval_report_<ver>.json (model_dump_json)
                                                          [cli/run_phase4.py]
```

## 10. 테스트 등급 및 완료 기준

| 모듈 | 등급([[CLAUDE.md]] 2절) | 목표 커버리지 | 비고 |
|---|---|---|---|
| `domain/models.py`(`ValidationMetrics`/`GapMetrics`/`TargetCheckResult`/`EvaluationReport` 추가분) | A | ≥ 90% | 순수 데이터 모델 |
| `evaluation/metrics.py` | A | ≥ 90% | Red→Green→Refactor로 테스트 먼저 작성. fixture `(y_true, y_pred)` 조합으로 accuracy/f1-macro/confusion matrix/이진 집계/gap/target 판정 각각 검증, `probs_to_labels` 동률 tie-break 케이스 포함 |
| `evaluation/report.py` | B | ≥ 70% | fake `Classifier`(고정 확률 반환)로 `build_report` 통합 테스트, `save_report` 기존 파일 존재 시 실패(md만 존재/json만 존재 두 케이스 모두) |
| `training/persistence.py`(`load_search_result` 추가분) | B | ≥ 70% | `tmp_path` 왕복 + 파일 부재 시 `ModelNotFoundError` |
| `cli/run_phase4.py` | B | ≥ 70% | fixture parquet + fixture model.pkl + fixture hyperparams.json + fixture config JSON으로 end-to-end 실행, 목표 미달 케이스에서도 exit code 0 확인 |

완료 기준: 위 5개 대상이 각 목표 커버리지를 충족하고, (1) `val_vectors.parquet`을 어떤
학습·재학습에도 사용하지 않음, (2) gap 계산이 `hyperparams.json`의 test 성적을 정확히
반영함, (3) 목표 미달 시에도 `cli/run_phase4.py`가 exit code 0으로 종료함, (4)
`evaluation/report.py`가 sklearn을 직접 import하지 않고 `Classifier` Protocol에만
의존함이 테스트로 확인되면 Phase 4 코드는 완료로 간주한다. 실측치는 이후 작성할
`P4_테스트결과서_Validation.md`에 기록한다.

## 11. 관련 문서/코드

- 요구사항: [[P4_요구사항정의서_Validation]]
- 상위 설계: [[Architecture_Design]] 2절(모듈 구조, `evaluation/*` earmark)
- 참조 패턴: [[P3_설계서_Training]] 4절(`domain/interfaces.Classifier` 재사용), 5절
  (`persistence.py` 저장/로드 대칭 패턴), 6절(JSON 설정 파일 + `--config` 패턴)
- 공통 모듈: `domain/interfaces.py`(`Classifier` Protocol, 갱신 없음),
  `exceptions.py`(`DataValidationError`/`ModelNotFoundError`, 갱신 없음)
- 관련 코드(신규): `src/embedding_lr/evaluation/{metrics,report}.py`,
  `src/embedding_lr/cli/run_phase4.py`, `config/eval_thresholds_default.json`
- 관련 코드(갱신): `src/embedding_lr/domain/models.py`, `src/embedding_lr/constants.py`,
  `src/embedding_lr/training/persistence.py`(`load_search_result` 추가)
- 테스트: `tests/unit/test_models.py`(갱신), `tests/unit/test_metrics.py`(신규),
  `tests/integration/test_{report,run_phase4}.py`(신규),
  `tests/integration/test_persistence.py`(`load_search_result` 케이스 추가)
- 신규 의존성: 없음(`scikit-learn`은 Phase 3에서 이미 추가됨, `pyproject.toml` 변경 불필요)
