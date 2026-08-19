import pytest
from pydantic import ValidationError

from embedding_lr.constants import CLASS_LABELS, EMBEDDING_DIM
from embedding_lr.domain.models import EmbeddingVector, PredictionResult, QueryRecord


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


class TestEmbeddingVector:
    def test_accepts_vector_with_correct_dimension(self):
        vector = EmbeddingVector(vector=[0.1] * EMBEDDING_DIM, category="IT")
        assert len(vector.vector) == EMBEDDING_DIM

    def test_rejects_vector_with_wrong_dimension(self):
        with pytest.raises(ValidationError):
            EmbeddingVector(vector=[0.1] * (EMBEDDING_DIM - 1), category="IT")

    def test_rejects_unknown_category(self):
        with pytest.raises(ValidationError):
            EmbeddingVector(vector=[0.1] * EMBEDDING_DIM, category="UNKNOWN")


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
