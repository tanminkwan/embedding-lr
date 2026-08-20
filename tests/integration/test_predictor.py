from embedding_lr.domain.models import QueryRecord
from embedding_lr.inference import predictor


class _FakeTextClassifier:
    def __init__(self, probs: list[dict[str, float]]) -> None:
        self._probs = probs
        self.calls: list[list[str]] = []

    def classify(self, queries: list[str]) -> list[dict[str, float]]:
        self.calls.append(queries)
        return self._probs


def _all_class_probs(it_weight: float) -> dict[str, float]:
    labels = ["IT", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"]
    remaining = (1 - it_weight) / (len(labels) - 1)
    return {label: (it_weight if label == "IT" else remaining) for label in labels}


class TestPredict:
    def test_returns_empty_list_without_calling_classifier_when_items_empty(self):
        classifier = _FakeTextClassifier(probs=[])

        result = predictor.predict(classifier, [])

        assert result == []
        assert classifier.calls == []

    def test_calls_classifier_exactly_once_with_cleaned_queries(self):
        probs = [_all_class_probs(0.9), _all_class_probs(0.05)]
        classifier = _FakeTextClassifier(probs)
        items = [
            QueryRecord(query="Tomcat  재시작?", response="아무거나"),
            QueryRecord(query="오늘 날씨 어때?", response="상관없는 응답"),
        ]

        results = predictor.predict(classifier, items)

        assert len(classifier.calls) == 1
        assert classifier.calls[0] == ["Tomcat 재시작?", "오늘 날씨 어때?"]
        assert len(results) == 2
        assert results[0].predicted_category == "IT"
        assert results[0].final_verdict == "IT"
        assert results[1].final_verdict == "NON_IT"

    def test_response_field_does_not_affect_result(self):
        probs = [_all_class_probs(0.9)]
        classifier_a = _FakeTextClassifier(probs)
        classifier_b = _FakeTextClassifier(probs)
        item_a = QueryRecord(query="Tomcat 재시작?", response="응답 A")
        item_b = QueryRecord(query="Tomcat 재시작?", response="완전히 다른 응답 B")

        result_a = predictor.predict(classifier_a, [item_a])
        result_b = predictor.predict(classifier_b, [item_b])

        assert classifier_a.calls == classifier_b.calls
        assert result_a[0].predicted_category == result_b[0].predicted_category

    def test_results_correspond_to_items_in_order(self):
        probs = [_all_class_probs(0.9), _all_class_probs(0.05)]
        classifier = _FakeTextClassifier(probs)
        items = [
            QueryRecord(query="IT 질문", response=""),
            QueryRecord(query="일상 질문", response=""),
        ]

        results = predictor.predict(classifier, items)

        assert results[0].predicted_category == "IT"
        assert results[1].predicted_category != "IT"
