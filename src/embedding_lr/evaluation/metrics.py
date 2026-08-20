"""검증 지표 순수 계산 — P4_설계서_Validation.md 4절. 등급 A(핵심 순수 로직) — 테스트 먼저.

Trigger: evaluation/report.py가 아래 함수들을 순서대로 호출.
Input:   (y_true, y_pred) 라벨 리스트, 또는 이미 계산된 ValidationMetrics/HyperparamSearchResult.
Output:  ValidationMetrics / GapMetrics / TargetCheckResult — 파일 I/O·모델 접근 없음.
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from embedding_lr.constants import (
    CLASS_LABELS,
    IT_LABEL,
    TARGET_ACCURACY,
    TARGET_BINARY_ACCURACY,
    TARGET_F1_MACRO,
)
from embedding_lr.domain.models import (
    GapMetrics,
    HyperparamSearchResult,
    TargetCheckResult,
    ValidationMetrics,
)

_BINARY_LABELS = ["IT", "NON_IT"]
_REPORT_FIELDS = {"precision": "precision", "recall": "recall", "f1-score": "f1", "support": "support"}


def probs_to_labels(probs: list[dict[str, float]]) -> list[str]:
    """각 레코드의 확률 dict에서 최댓값 라벨을 선택. 동률 시 CLASS_LABELS 순서상 먼저
    나오는 라벨을 선택한다(결정적 tie-break, 재현성 보장). 모델이 일부 클래스만 학습해
    dict에 CLASS_LABELS 전부가 없는 경우에도 존재하는 라벨만으로 판단한다."""
    return [
        max((label for label in CLASS_LABELS if label in row), key=lambda label: row[label])
        for row in probs
    ]


def to_binary_labels(labels: list[str]) -> list[str]:
    """IT_LABEL이면 "IT", 그 외(NON_IT_LABELS)면 "NON_IT"로 매핑."""
    return [IT_LABEL if label == IT_LABEL else "NON_IT" for label in labels]


def compute_metrics(y_true: list[str], y_pred: list[str]) -> ValidationMetrics:
    """5-class Accuracy/F1-macro/Confusion Matrix/Classification Report와, 이진(IT vs
    NON_IT) 집계 Accuracy/Confusion Matrix를 함께 계산한다."""
    y_true_bin = to_binary_labels(y_true)
    y_pred_bin = to_binary_labels(y_pred)

    raw_report = classification_report(
        y_true, y_pred, labels=CLASS_LABELS, output_dict=True, zero_division=0
    )
    report = {
        label: {out_key: float(raw_report[label][in_key]) for in_key, out_key in _REPORT_FIELDS.items()}
        for label in CLASS_LABELS
    }

    return ValidationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        binary_accuracy=float(accuracy_score(y_true_bin, y_pred_bin)),
        confusion_matrix_labels=list(CLASS_LABELS),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
        binary_confusion_matrix=confusion_matrix(y_true_bin, y_pred_bin, labels=_BINARY_LABELS).tolist(),
        classification_report=report,
    )


def compute_gap(
    search_result: HyperparamSearchResult, val_metrics: ValidationMetrics, gap_warning_threshold: float,
) -> GapMetrics:
    """test set 성적(search_result)과 val set 성적(val_metrics) 간 차이를 계산한다.
    둘 중 하나라도 gap_warning_threshold를 초과하면 warning=True."""
    accuracy_gap = search_result.best_accuracy - val_metrics.accuracy
    f1_macro_gap = search_result.best_f1_macro - val_metrics.f1_macro
    warning = accuracy_gap > gap_warning_threshold or f1_macro_gap > gap_warning_threshold
    return GapMetrics(accuracy_gap=accuracy_gap, f1_macro_gap=f1_macro_gap, warning=warning)


def check_targets(val_metrics: ValidationMetrics) -> TargetCheckResult:
    """val_metrics를 Scope_Definition 4.4절 목표치(constants.py)와 비교(>=)."""
    return TargetCheckResult(
        accuracy_target_met=val_metrics.accuracy >= TARGET_ACCURACY,
        binary_accuracy_target_met=val_metrics.binary_accuracy >= TARGET_BINARY_ACCURACY,
        f1_macro_target_met=val_metrics.f1_macro >= TARGET_F1_MACRO,
    )
