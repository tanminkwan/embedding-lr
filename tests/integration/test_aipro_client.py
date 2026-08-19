import json

import httpx
import pytest
import respx

from embedding_lr.config import Settings
from embedding_lr.domain.models import KnowledgeRecord
from embedding_lr.embedding.aipro_client import AIProClient
from embedding_lr.exceptions import AIProClientError


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        aipro_base_url="http://localhost:28000",
        aipro_api_token="test-token",
        embedding_server_base_url="http://localhost:8000",
        model_dir="models",
    )


def _knowledge_item(**overrides) -> dict:
    item = {
        "id": "1",
        "collection": "v0.2_train",
        "content": "query text",
        "extended_content": "query text\nresponse text",
        "domain_id": 1,
        "source": "IT",
        "created_at": "2026-08-19T00:00:00Z",
        "embedding": [0.1] * 1024,
    }
    item.update(overrides)
    return item


class TestGetKnowledge:
    @respx.mock
    def test_returns_knowledge_items_including_embedding(self):
        respx.get("http://localhost:28000/api/rag/knowledge").mock(
            return_value=httpx.Response(200, json=[_knowledge_item()])
        )
        client = AIProClient(_settings())

        items = client.get_knowledge(domain_id=1, collection="v0.2_train", limit=1000)

        assert len(items) == 1
        assert items[0].embedding == [0.1] * 1024
        assert items[0].source == "IT"

    @respx.mock
    def test_sends_domain_id_collection_and_limit_as_query_params(self):
        route = respx.get("http://localhost:28000/api/rag/knowledge").mock(
            return_value=httpx.Response(200, json=[])
        )
        client = AIProClient(_settings())

        client.get_knowledge(domain_id=1, collection="v0.2_train", limit=1000)

        request = route.calls.last.request
        assert request.url.params["domain_id"] == "1"
        assert request.url.params["collection"] == "v0.2_train"
        assert request.url.params["limit"] == "1000"

    @respx.mock
    def test_returns_empty_list_when_collection_has_no_items(self):
        respx.get("http://localhost:28000/api/rag/knowledge").mock(
            return_value=httpx.Response(200, json=[])
        )
        client = AIProClient(_settings())

        assert client.get_knowledge(domain_id=1, collection="v0.2_train", limit=1000) == []

    @respx.mock
    def test_raises_aipro_client_error_on_http_error_status(self):
        respx.get("http://localhost:28000/api/rag/knowledge").mock(
            return_value=httpx.Response(500, json={"detail": "internal error"})
        )
        client = AIProClient(_settings())

        with pytest.raises(AIProClientError, match="GET /api/rag/knowledge 실패"):
            client.get_knowledge(domain_id=1, collection="v0.2_train", limit=1000)

    @respx.mock
    def test_raises_aipro_client_error_on_malformed_response_schema(self):
        respx.get("http://localhost:28000/api/rag/knowledge").mock(
            return_value=httpx.Response(200, json=[{"id": "1"}])
        )
        client = AIProClient(_settings())

        with pytest.raises(AIProClientError, match="GET /api/rag/knowledge 응답 파싱 실패"):
            client.get_knowledge(domain_id=1, collection="v0.2_train", limit=1000)


class TestUpsert:
    @respx.mock
    def test_posts_each_record_individually_not_bulk_upload(self):
        route = respx.post("http://localhost:28000/api/rag/knowledge").mock(
            return_value=httpx.Response(200, json={})
        )
        client = AIProClient(_settings())
        records = [
            KnowledgeRecord(content="q1", extended_content="q1\nr1", source="IT"),
            KnowledgeRecord(content="q2", extended_content="q2\nr2", source="DAILY"),
        ]

        client.upsert(records, domain_id=1, collection="v0_2_train")

        assert route.call_count == 2

    @respx.mock
    def test_sends_content_source_domain_id_and_collection_name_as_payload(self):
        route = respx.post("http://localhost:28000/api/rag/knowledge").mock(
            return_value=httpx.Response(200, json={})
        )
        client = AIProClient(_settings())
        records = [KnowledgeRecord(content="q1", extended_content="q1\nr1", source="IT")]

        client.upsert(records, domain_id=1, collection="v0_2_train")

        payload = json.loads(route.calls.last.request.content)
        assert payload == {
            "content": "q1",
            "extended_content": "q1\nr1",
            "source": "IT",
            "domain_id": 1,
            "collection_name": "v0_2_train",
        }

    @respx.mock
    def test_raises_aipro_client_error_on_http_error_status(self):
        respx.post("http://localhost:28000/api/rag/knowledge").mock(
            return_value=httpx.Response(500, json={"detail": "internal error"})
        )
        client = AIProClient(_settings())
        records = [KnowledgeRecord(content="q1", extended_content="q1\nr1", source="IT")]

        with pytest.raises(AIProClientError, match="POST /api/rag/knowledge 실패"):
            client.upsert(records, domain_id=1, collection="v0_2_train")


class TestClose:
    def test_close_closes_underlying_http_client(self):
        client = AIProClient(_settings())

        client.close()

        assert client._client.is_closed


class TestAuthHeader:
    def test_sends_bearer_header_when_token_is_set(self):
        client = AIProClient(_settings())

        assert client._client.headers["Authorization"] == "Bearer test-token"

    def test_omits_authorization_header_when_token_is_empty(self):
        settings = _settings()
        settings.aipro_api_token = ""

        client = AIProClient(settings)

        assert "Authorization" not in client._client.headers
