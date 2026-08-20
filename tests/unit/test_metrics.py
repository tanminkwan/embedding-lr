import pytest

from embedding_lr.domain.models import HyperparamSearchResult, HyperparamTrial, ValidationMetrics
from embedding_lr.evaluation import metrics


def _val_metrics(**overrides) -> ValidationMetrics:
    payload = dict(
        accuracy=0.8,
        f1_macro=0.78,
        binary_accuracy=0.9,
        confusion_matrix_labels=["IT", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"],
        confusion_matrix=[[1, 0, 0, 0, 0]] * 5,
        binary_confusion_matrix=[[1, 0], [0, 1]],
        classification_report={},
    )
    payload.update(overrides)
    return ValidationMetrics(**payload)


class TestProbsToLabels:
    def test_returns_argmax_label_per_row(self):
        probs = [
            {"IT": 0.7, "DAILY": 0.1, "KNOWLEDGE": 0.1, "CREATIVE": 0.05, "ANOMALY": 0.05},
            {"IT": 0.1, "DAILY": 0.6, "KNOWLEDGE": 0.1, "CREATIVE": 0.1, "ANOMALY": 0.1},
        ]

        labels = metrics.probs_to_labels(probs)

        assert labels == ["IT", "DAILY"]

    def test_breaks_ties_using_class_labels_order(self):
        # CLASS_LABELS 순서: IT, DAILY, KNOWLEDGE, CREATIVE, ANOMALY — KNOWLEDGE가 IT/DAILY보다 뒤라
        # 동률이면 IT가 선택되어야 한다(결정적 tie-break, 재현성 보장).
        probs = [{"IT": 0.5, "DAILY": 0.5, "KNOWLEDGE": 0.0, "CREATIVE": 0.0, "ANOMALY": 0.0}]

        labels = metrics.probs_to_labels(probs)

        assert labels == ["IT"]

    def test_handles_dicts_missing_some_class_labels(self):
        # 모델이 일부 클래스만 학습한 경우(predict_proba 키가 CLASS_LABELS 전체가 아님).
        probs = [{"IT": 0.4, "DAILY": 0.6}]

        labels = metrics.probs_to_labels(probs)

        assert labels == ["DAILY"]


class TestToBinaryLabels:
    def test_maps_it_and_non_it_labels(self):
        labels = ["IT", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"]

        binary = metrics.to_binary_labels(labels)

        assert binary == ["IT", "NON_IT", "NON_IT", "NON_IT", "NON_IT"]


class TestComputeMetrics:
    def test_computes_perfect_score_for_identical_labels(self):
        y_true = ["IT", "IT", "DAILY", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"]
        y_pred = list(y_true)

        result = metrics.compute_metrics(y_true, y_pred)

        assert result.accuracy == pytest.approx(1.0)
        assert result.f1_macro == pytest.approx(1.0)
        assert result.binary_accuracy == pytest.approx(1.0)
        assert result.confusion_matrix_labels == ["IT", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"]
        assert sum(sum(row) for row in result.confusion_matrix) == len(y_true)
        assert sum(sum(row) for row in result.binary_confusion_matrix) == len(y_true)

    def test_classification_report_has_entry_per_class_label(self):
        y_true = ["IT", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"]
        y_pred = ["IT", "IT", "KNOWLEDGE", "CREATIVE", "ANOMALY"]

        result = metrics.compute_metrics(y_true, y_pred)

        assert set(result.classification_report.keys()) == {
            "IT", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY",
        }
        assert set(result.classification_report["IT"].keys()) == {
            "precision", "recall", "f1", "support",
        }

    def test_misclassification_lowers_accuracy_and_binary_accuracy(self):
        y_true = ["IT", "IT", "DAILY", "DAILY"]
        y_pred = ["DAILY", "IT", "DAILY", "DAILY"]

        result = metrics.compute_metrics(y_true, y_pred)

        assert result.accuracy == pytest.approx(0.75)
        assert result.binary_accuracy == pytest.approx(0.75)


class TestComputeGap:
    def test_computes_gap_without_warning_below_threshold(self):
        search_result = HyperparamSearchResult(
            best_params={"C": 1.0}, best_accuracy=0.9, best_f1_macro=0.88, trials=[]
        )
        val_metrics = _val_metrics(accuracy=0.87, f1_macro=0.85)

        gap = metrics.compute_gap(search_result, val_metrics, gap_warning_threshold=0.1)

        assert gap.accuracy_gap == pytest.approx(0.03)
        assert gap.f1_macro_gap == pytest.approx(0.03)
        assert gap.warning is False

    def test_flags_warning_when_accuracy_gap_exceeds_threshold(self):
        search_result = HyperparamSearchResult(
            best_params={"C": 1.0}, best_accuracy=0.95, best_f1_macro=0.9, trials=[]
        )
        val_metrics = _val_metrics(accuracy=0.7, f1_macro=0.85)

        gap = metrics.compute_gap(search_result, val_metrics, gap_warning_threshold=0.1)

        assert gap.accuracy_gap == pytest.approx(0.25)
        assert gap.warning is True

    def test_flags_warning_when_f1_macro_gap_exceeds_threshold(self):
        search_result = HyperparamSearchResult(
            best_params={"C": 1.0}, best_accuracy=0.9, best_f1_macro=0.95, trials=[]
        )
        val_metrics = _val_metrics(accuracy=0.88, f1_macro=0.6)

        gap = metrics.compute_gap(search_result, val_metrics, gap_warning_threshold=0.1)

        assert gap.f1_macro_gap == pytest.approx(0.35)
        assert gap.warning is True


class TestCheckTargets:
    def test_all_targets_met(self):
        val_metrics = _val_metrics(accuracy=0.9, binary_accuracy=0.95, f1_macro=0.9)

        result = metrics.check_targets(val_metrics)

        assert result.accuracy_target_met is True
        assert result.binary_accuracy_target_met is True
        assert result.f1_macro_target_met is True

    def test_reports_each_target_independently(self):
        val_metrics = _val_metrics(accuracy=0.5, binary_accuracy=0.95, f1_macro=0.5)

        result = metrics.check_targets(val_metrics)

        assert result.accuracy_target_met is False
        assert result.binary_accuracy_target_met is True
        assert result.f1_macro_target_met is False
