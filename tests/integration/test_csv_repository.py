import pytest

from embedding_lr.data_generation.csv_repository import CsvRepository
from embedding_lr.domain.models import QueryRecord
from embedding_lr.exceptions import DataValidationError


class TestCsvRepository:
    def test_loads_valid_csv_rows(self, tmp_path):
        path = tmp_path / "role_01_middleware.csv"
        path.write_text(
            "질의,응답,카테고리\n"
            'nginx 재시작?,systemctl restart nginx,IT\n'
            '오늘 날씨 어때?,맑음,DAILY\n',
            encoding="utf-8",
        )

        records = CsvRepository().load(str(path))

        assert records == [
            QueryRecord(query="nginx 재시작?", response="systemctl restart nginx", category="IT"),
            QueryRecord(query="오늘 날씨 어때?", response="맑음", category="DAILY"),
        ]

    def test_handles_embedded_newline_in_quoted_field(self, tmp_path):
        path = tmp_path / "role_03_network.csv"
        path.write_text(
            '질의,응답,카테고리\n'
            '"방화벽 로그 확인 순서?","확인 순서:\n1. IP 필터링\n2. 정책 확인",IT\n',
            encoding="utf-8",
        )

        records = CsvRepository().load(str(path))

        assert len(records) == 1
        assert "확인 순서:\n1. IP 필터링\n2. 정책 확인" == records[0].response

    def test_raises_on_missing_column(self, tmp_path):
        path = tmp_path / "broken.csv"
        path.write_text("질의,카테고리\nq,IT\n", encoding="utf-8")

        with pytest.raises(DataValidationError, match="필수 컬럼 누락"):
            CsvRepository().load(str(path))

    def test_raises_on_unknown_category(self, tmp_path):
        path = tmp_path / "bad_category.csv"
        path.write_text("질의,응답,카테고리\nq,r,UNKNOWN\n", encoding="utf-8")

        with pytest.raises(DataValidationError, match="필드 검증 실패"):
            CsvRepository().load(str(path))

    def test_save_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            CsvRepository().save([], "irrelevant.csv")
