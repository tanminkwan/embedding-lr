from embedding_lr.inference.embedding_lr_classifier import EmbeddingLRTextClassifier


class _FakeEmbeddingClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self._vectors


class _FakeClassifier:
    def __init__(self, probs: list[dict[str, float]]) -> None:
        self._probs = probs
        self.calls: list[list[list[float]]] = []

    def fit(self, X, y) -> None:  # pragma: no cover - Classifier Protocol 준수용
        raise NotImplementedError

    def predict_proba(self, X: list[list[float]]) -> list[dict[str, float]]:
        self.calls.append(X)
        return self._probs


class TestClassify:
    def test_returns_empty_list_without_calling_dependencies_when_queries_empty(self):
        embedding_client = _FakeEmbeddingClient(vectors=[])
        model = _FakeClassifier(probs=[])
        classifier = EmbeddingLRTextClassifier(embedding_client, model)

        result = classifier.classify([])

        assert result == []
        assert embedding_client.calls == []
        assert model.calls == []

    def test_embeds_then_predicts_exactly_once(self):
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        probs = [{"IT": 0.9, "DAILY": 0.1}, {"IT": 0.2, "DAILY": 0.8}]
        embedding_client = _FakeEmbeddingClient(vectors)
        model = _FakeClassifier(probs)
        classifier = EmbeddingLRTextClassifier(embedding_client, model)

        result = classifier.classify(["질의 1", "질의 2"])

        assert result == probs
        assert embedding_client.calls == [["질의 1", "질의 2"]]
        assert model.calls == [vectors]
