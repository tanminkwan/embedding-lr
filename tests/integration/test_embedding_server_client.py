import json

import httpx
import pytest
import respx

from embedding_lr.config import Settings
from embedding_lr.embedding.embedding_server_client import EmbeddingServerClient
from embedding_lr.exceptions import EmbeddingServerError


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        aipro_base_url="http://localhost:28000",
        aipro_api_token="test-token",
        embedding_server_base_url="http://localhost:8000",
        model_dir="models",
        model_path="models/model.pkl",
    )


class TestEmbed:
    @respx.mock
    def test_returns_embeddings_for_each_text(self):
        respx.post("http://localhost:8000/embed").mock(
            return_value=httpx.Response(
                200,
                json={"embeddings": [[0.1] * 1024, [0.2] * 1024], "dim": 1024, "count": 2},
            )
        )
        client = EmbeddingServerClient(_settings())

        vectors = client.embed(["query 1", "query 2"])

        assert vectors == [[0.1] * 1024, [0.2] * 1024]

    @respx.mock
    def test_sends_texts_in_request_body(self):
        route = respx.post("http://localhost:8000/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [], "dim": 1024, "count": 0})
        )
        client = EmbeddingServerClient(_settings())

        client.embed(["query 1"])

        request = route.calls.last.request
        assert json.loads(request.content) == {"texts": ["query 1"]}

    @respx.mock
    def test_raises_embedding_server_error_on_http_error_status(self):
        respx.post("http://localhost:8000/embed").mock(
            return_value=httpx.Response(500, json={"detail": "internal error"})
        )
        client = EmbeddingServerClient(_settings())

        with pytest.raises(EmbeddingServerError, match="POST /embed 실패"):
            client.embed(["query 1"])

    @respx.mock
    def test_raises_embedding_server_error_on_malformed_response_schema(self):
        respx.post("http://localhost:8000/embed").mock(
            return_value=httpx.Response(200, json={"unexpected": "shape"})
        )
        client = EmbeddingServerClient(_settings())

        with pytest.raises(EmbeddingServerError, match="POST /embed 응답 파싱 실패"):
            client.embed(["query 1"])


class TestClose:
    def test_close_closes_underlying_http_client(self):
        client = EmbeddingServerClient(_settings())

        client.close()

        assert client._client.is_closed
