import pytest
from pydantic import ValidationError

from embedding_lr.constants import CLASS_LABELS, EMBEDDING_DIM
from embedding_lr.domain.models import (
    EvaluationReport,
    GapMetrics,
    HyperparamSearchResult,
    HyperparamTrial,
    KnowledgeItem,
    KnowledgeRecord,
    PredictionResult,
    QueryRecord,
    TargetCheckResult,
    ValidationMetrics,
)


class TestQueryRecord:
    def test_accepts_known_category(self):
        record = QueryRecord(query="Tomcat 재시작?", response="shutdown 후 startup", category="IT")
        assert record.category == "IT"

    def test_allows_missing_category_for_inference_request(self):
        record = QueryRecord(query="오늘 날씨 어때?", response="")
        assert record.category is None

    def test_rejects_unknown_category(self):
        with pytest.raises(ValidationError):
            QueryRecord(query="q", response="r", category="UNKNOWN")


class TestKnowledgeRecord:
    def test_accepts_known_source_label(self):
        record = KnowledgeRecord(content="Tomcat 재시작?", extended_content="Tomcat 재시작? / shutdown 후 startup", source="IT")
        assert record.source == "IT"

    def test_rejects_unknown_source_label(self):
        with pytest.raises(ValidationError):
            KnowledgeRecord(content="q", extended_content="q / r", source="UNKNOWN")


class TestKnowledgeItem:
    def _item(self, **overrides) -> dict:
        item = dict(
            id="1",
            collection="v0.2_train",
            content="query text",
            extended_content="query text / response text",
            domain_id=1,
            source="IT",
            created_at="2026-08-19T00:00:00Z",
            embedding=[0.1] * EMBEDDING_DIM,
        )
        item.update(overrides)
        return item

    def test_accepts_embedding_with_correct_dimension(self):
        item = KnowledgeItem(**self._item())
        assert len(item.embedding) == EMBEDDING_DIM

    def test_rejects_embedding_with_wrong_dimension(self):
        with pytest.raises(ValidationError):
            KnowledgeItem(**self._item(embedding=[0.1] * (EMBEDDING_DIM - 1)))


class TestPredictionResult:
    def _probabilities(self, it_weight: float = 0.9) -> dict[str, float]:
        remaining = (1 - it_weight) / (len(CLASS_LABELS) - 1)
        return {label: (it_weight if label == "IT" else remaining) for label in CLASS_LABELS}

    def test_accepts_consistent_it_verdict(self):
        result = PredictionResult(
            predicted_category="IT", final_verdict="IT", probabilities=self._probabilities()
        )
        assert result.final_verdict == "IT"

    def test_accepts_consistent_non_it_verdict(self):
        probs = self._probabilities(it_weight=0.05)
        result = PredictionResult(
            predicted_category="DAILY", final_verdict="NON_IT", probabilities=probs
        )
        assert result.final_verdict == "NON_IT"

    def test_rejects_inconsistent_verdict(self):
        with pytest.raises(ValidationError):
            PredictionResult(
                predicted_category="IT", final_verdict="NON_IT", probabilities=self._probabilities()
            )

    def test_rejects_unknown_final_verdict_literal(self):
        with pytest.raises(ValidationError):
            PredictionResult(
                predicted_category="IT", final_verdict="MAYBE", probabilities=self._probabilities()
            )

    def test_rejects_unknown_predicted_category(self):
        with pytest.raises(ValidationError):
            PredictionResult(
                predicted_category="UNKNOWN", final_verdict="NON_IT", probabilities=self._probabilities()
            )

    def test_rejects_probabilities_missing_a_class_label(self):
        incomplete = self._probabilities()
        del incomplete["ANOMALY"]
        with pytest.raises(ValidationError):
            PredictionResult(predicted_category="IT", final_verdict="IT", probabilities=incomplete)

    def test_rejects_probabilities_with_extra_key(self):
        extra = self._probabilities()
        extra["EXTRA"] = 0.0
        with pytest.raises(ValidationError):
            PredictionResult(predicted_category="IT", final_verdict="IT", probabilities=extra)


class TestHyperparamTrial:
    def test_accepts_mixed_type_params(self):
        trial = HyperparamTrial(
            params={"C": 1.0, "solver": "lbfgs", "max_iter": 500}, accuracy=0.9, f1_macro=0.88
        )
        assert trial.params["solver"] == "lbfgs"
        assert trial.accuracy == pytest.approx(0.9)

    def test_rejects_missing_field(self):
        with pytest.raises(ValidationError):
            HyperparamTrial(params={"C": 1.0}, accuracy=0.9)


class TestHyperparamSearchResult:
    def test_accepts_result_with_trials(self):
        trials = [
            HyperparamTrial(params={"C": 0.1}, accuracy=0.8, f1_macro=0.78),
            HyperparamTrial(params={"C": 1.0}, accuracy=0.9, f1_macro=0.88),
        ]
        result = HyperparamSearchResult(
            best_params={"C": 1.0}, best_accuracy=0.9, best_f1_macro=0.88, trials=trials
        )
        assert len(result.trials) == 2
        assert result.best_params == {"C": 1.0}

    def test_accepts_empty_trials_list(self):
        result = HyperparamSearchResult(
            best_params={"C": 1.0}, best_accuracy=0.9, best_f1_macro=0.88, trials=[]
        )
        assert result.trials == []

    def test_rejects_missing_field(self):
        with pytest.raises(ValidationError):
            HyperparamSearchResult(best_accuracy=0.9, best_f1_macro=0.88, trials=[])


def _validation_metrics_payload(**overrides) -> dict:
    metrics = dict(
        accuracy=0.9,
        f1_macro=0.88,
        binary_accuracy=0.95,
        confusion_matrix_labels=CLASS_LABELS,
        confusion_matrix=[[1] * len(CLASS_LABELS) for _ in CLASS_LABELS],
        binary_confusion_matrix=[[1, 0], [0, 1]],
        classification_report={"IT": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 40}},
    )
    metrics.update(overrides)
    return metrics


class TestValidationMetrics:
    def test_accepts_full_metrics(self):
        metrics = ValidationMetrics(**_validation_metrics_payload())
        assert metrics.accuracy == pytest.approx(0.9)
        assert len(metrics.confusion_matrix) == len(CLASS_LABELS)

    def test_rejects_missing_field(self):
        payload = _validation_metrics_payload()
        del payload["f1_macro"]
        with pytest.raises(ValidationError):
            ValidationMetrics(**payload)


class TestGapMetrics:
    def test_accepts_warning_flag(self):
        gap = GapMetrics(accuracy_gap=0.12, f1_macro_gap=0.05, warning=True)
        assert gap.warning is True

    def test_rejects_missing_field(self):
        with pytest.raises(ValidationError):
            GapMetrics(accuracy_gap=0.12, warning=True)


class TestTargetCheckResult:
    def test_accepts_all_targets_met(self):
        result = TargetCheckResult(
            accuracy_target_met=True, binary_accuracy_target_met=True, f1_macro_target_met=True
        )
        assert result.accuracy_target_met is True

    def test_rejects_missing_field(self):
        with pytest.raises(ValidationError):
            TargetCheckResult(accuracy_target_met=True, binary_accuracy_target_met=True)


class TestEvaluationReport:
    def test_accepts_nested_models(self):
        report = EvaluationReport(
            metrics=ValidationMetrics(**_validation_metrics_payload()),
            gap=GapMetrics(accuracy_gap=0.02, f1_macro_gap=0.01, warning=False),
            targets=TargetCheckResult(
                accuracy_target_met=True, binary_accuracy_target_met=True, f1_macro_target_met=True
            ),
        )
        assert report.gap.warning is False

    def test_rejects_missing_field(self):
        with pytest.raises(ValidationError):
            EvaluationReport(
                gap=GapMetrics(accuracy_gap=0.02, f1_macro_gap=0.01, warning=False),
                targets=TargetCheckResult(
                    accuracy_target_met=True, binary_accuracy_target_met=True, f1_macro_target_met=True
                ),
            )
