"""Phase 4 검증 오케스트레이션 — P4_설계서_Validation.md 5절. 등급 B(오케스트레이션) —
`Classifier` Protocol(domain/interfaces.py)에만 의존, sklearn을 직접 import하지 않는다.

Trigger: build_report(...) → save_report(...) — cli/run_phase4.py가 감싼다.
Input:   Classifier 구현체, val set(X, y), HyperparamSearchResult, gap 경고 임계값
Output:  EvaluationReport / eval_report_<ver>.md, .json (저장 시 경로가 이미 존재하면 실패)
"""

from pathlib import Path

from embedding_lr.domain.interfaces import Classifier
from embedding_lr.domain.models import EvaluationReport, HyperparamSearchResult
from embedding_lr.evaluation import metrics
from embedding_lr.exceptions import DataValidationError

_GAP_GUIDE = (
    "gap(test 성적 - val 성적)이 크면서 목표 미달이면 Phase 3에서 test set에 과적합되도록 "
    "하이퍼파라미터가 선택되었을 가능성을 의심하고(→ Phase 3 재작업), gap이 작은데도 목표 "
    "미달이면 데이터 자체(라벨 품질, 클래스 경계 모호성)를 의심한다(→ Phase 1 재작업). "
    "루프백 실행 여부는 이 리포트를 본 사람이 최종 판단한다."
)


def build_report(
    model: Classifier,
    X_val: list[list[float]],
    y_val: list[str],
    search_result: HyperparamSearchResult,
    gap_warning_threshold: float,
) -> EvaluationReport:
    """model.predict_proba(X_val)로 val set을 예측하고, 5-class/이진 지표·gap·목표
    달성 여부를 계산해 EvaluationReport로 조립한다."""
    probs = model.predict_proba(X_val)
    y_pred = metrics.probs_to_labels(probs)

    val_metrics = metrics.compute_metrics(y_val, y_pred)
    gap = metrics.compute_gap(search_result, val_metrics, gap_warning_threshold)
    targets = metrics.check_targets(val_metrics)

    return EvaluationReport(metrics=val_metrics, gap=gap, targets=targets)


def _matrix_table(labels: list[str], matrix: list[list[int]]) -> str:
    header = "| 실제\\예측 | " + " | ".join(labels) + " |"
    separator = "|---" * (len(labels) + 1) + "|"
    rows = [
        f"| {label} | " + " | ".join(str(count) for count in row) + " |"
        for label, row in zip(labels, matrix)
    ]
    return "\n".join([header, separator, *rows])


def render_markdown(report: EvaluationReport) -> str:
    """EvaluationReport를 사람이 읽는 마크다운 리포트로 렌더링."""
    m, gap, targets = report.metrics, report.gap, report.targets

    lines = [
        "# Phase 4 검증 결과",
        "",
        "## 1. 지표 요약",
        "",
        "| 지표 | 값 | 목표 | 달성 |",
        "|---|---|---|---|",
        f"| 5-class Accuracy | {m.accuracy:.4f} | ≥0.85 | {'O' if targets.accuracy_target_met else 'X'} |",
        f"| IT vs NON_IT Accuracy | {m.binary_accuracy:.4f} | ≥0.90 | {'O' if targets.binary_accuracy_target_met else 'X'} |",
        f"| F1-macro | {m.f1_macro:.4f} | ≥0.85 | {'O' if targets.f1_macro_target_met else 'X'} |",
        "",
        "## 2. Test-vs-Validation Gap",
        "",
        f"{'⚠ ' if gap.warning else ''}Accuracy gap: {gap.accuracy_gap:.4f} / F1-macro gap: {gap.f1_macro_gap:.4f}",
        "",
        _GAP_GUIDE,
        "",
        "## 3. Confusion Matrix (5-class)",
        "",
        _matrix_table(m.confusion_matrix_labels, m.confusion_matrix),
        "",
        "## 4. Confusion Matrix (IT vs NON_IT)",
        "",
        _matrix_table(["IT", "NON_IT"], m.binary_confusion_matrix),
        "",
        "## 5. Classification Report",
        "",
        "| 클래스 | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for label in m.confusion_matrix_labels:
        entry = m.classification_report[label]
        lines.append(
            f"| {label} | {entry['precision']:.4f} | {entry['recall']:.4f} | "
            f"{entry['f1']:.4f} | {entry['support']:.0f} |"
        )
    lines.append("")
    return "\n".join(lines)


def save_report(report: EvaluationReport, md_path: str, json_path: str) -> None:
    """render_markdown(report)를 md_path에, report.model_dump_json(indent=2)를
    json_path에 저장. 둘 중 하나라도 이미 존재하면 아무것도 쓰지 않고 DataValidationError."""
    existing = [path for path in (md_path, json_path) if Path(path).exists()]
    if existing:
        raise DataValidationError(f"{existing} 이미 존재 — 덮어쓰기 금지(입출력 보존 원칙)")

    Path(md_path).write_text(render_markdown(report), encoding="utf-8")
    Path(json_path).write_text(report.model_dump_json(indent=2), encoding="utf-8")
