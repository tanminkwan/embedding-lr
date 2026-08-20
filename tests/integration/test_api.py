from fastapi.testclient import TestClient

from embedding_lr.exceptions import EmbeddingServerError
from embedding_lr.inference.api import create_app


class _FakeTextClassifier:
    def __init__(self, probs=None, error: Exception | None = None) -> None:
        self._probs = probs
        self._error = error
        self.calls: list[list[str]] = []

    def classify(self, queries: list[str]) -> list[dict[str, float]]:
        self.calls.append(queries)
        if self._error is not None:
            raise self._error
        return self._probs


def _all_class_probs(it_weight: float) -> dict[str, float]:
    labels = ["IT", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"]
    remaining = (1 - it_weight) / (len(labels) - 1)
    return {label: (it_weight if label == "IT" else remaining) for label in labels}


class TestClassifyEndpoint:
    def test_returns_predictions_for_items(self):
        classifier = _FakeTextClassifier(probs=[_all_class_probs(0.9)])
        client = TestClient(create_app(classifier))

        response = client.post(
            "/classify", json={"items": [{"query": "Tomcat 재시작?", "response": ""}]}
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 1
        assert body["results"][0]["predicted_category"] == "IT"

    def test_returns_empty_results_for_empty_items(self):
        classifier = _FakeTextClassifier(probs=[])
        client = TestClient(create_app(classifier))

        response = client.post("/classify", json={"items": []})

        assert response.status_code == 200
        assert response.json() == {"results": []}
        assert classifier.calls == []

    def test_returns_503_when_embedding_server_error_raised(self):
        classifier = _FakeTextClassifier(error=EmbeddingServerError("POST /embed 실패: boom"))
        client = TestClient(create_app(classifier))

        response = client.post(
            "/classify", json={"items": [{"query": "Tomcat 재시작?", "response": ""}]}
        )

        assert response.status_code == 503
        assert "boom" in response.json()["detail"]


class TestHealthEndpoint:
    def test_returns_ok(self):
        classifier = _FakeTextClassifier(probs=[])
        client = TestClient(create_app(classifier))

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
