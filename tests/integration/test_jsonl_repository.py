import pytest

from embedding_lr.data_generation.jsonl_repository import JsonlRepository
from embedding_lr.domain.models import QueryRecord
from embedding_lr.exceptions import DataValidationError


class TestJsonlRepository:
    def test_save_then_load_roundtrip(self, tmp_path):
        repo = JsonlRepository()
        path = str(tmp_path / "role_01_middleware.jsonl")
        records = [
            QueryRecord(query="nginx 재시작?", response="systemctl restart nginx", category="IT"),
            QueryRecord(query="오늘 날씨 어때?", response="맑음", category="DAILY"),
        ]

        repo.save(records, path)
        loaded = repo.load(path)

        assert loaded == records

    def test_save_raises_if_file_already_exists(self, tmp_path):
        repo = JsonlRepository()
        path = str(tmp_path / "data.jsonl")
        records = [QueryRecord(query="q", response="r", category="IT")]
        repo.save(records, path)

        with pytest.raises(DataValidationError, match="이미 존재"):
            repo.save(records, path)

    def test_load_raises_on_invalid_json_line(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text('{"질의": "q", "응답": "r", "카테고리": "IT"}\nnot json\n', encoding="utf-8")

        with pytest.raises(DataValidationError, match="JSON 파싱 실패"):
            JsonlRepository().load(str(path))

    def test_load_raises_on_missing_key(self, tmp_path):
        path = tmp_path / "missing_key.jsonl"
        path.write_text('{"질의": "q", "카테고리": "IT"}\n', encoding="utf-8")

        with pytest.raises(DataValidationError, match="필수 키 누락"):
            JsonlRepository().load(str(path))

    def test_load_raises_on_unknown_category(self, tmp_path):
        path = tmp_path / "bad_category.jsonl"
        path.write_text('{"질의": "q", "응답": "r", "카테고리": "UNKNOWN"}\n', encoding="utf-8")

        with pytest.raises(DataValidationError, match="필드 검증 실패"):
            JsonlRepository().load(str(path))

    def test_load_skips_blank_lines(self, tmp_path):
        path = tmp_path / "with_blank.jsonl"
        path.write_text('{"질의": "q", "응답": "r", "카테고리": "IT"}\n\n', encoding="utf-8")

        records = JsonlRepository().load(str(path))

        assert len(records) == 1
