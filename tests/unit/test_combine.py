import pytest

from embedding_lr.constants import RECORDS_PER_CLASS
from embedding_lr.dataset.combine import combine
from embedding_lr.domain.models import QueryRecord
from embedding_lr.exceptions import DataValidationError


def _records(category: str, count: int, query_prefix: str = "q") -> list[QueryRecord]:
    return [
        QueryRecord(query=f"{query_prefix}{i}", response=f"r{i}", category=category)
        for i in range(count)
    ]


class TestCombine:
    def test_combines_role_lists_when_class_counts_correct(self):
        role_records = [
            _records("IT", RECORDS_PER_CLASS),
            _records("DAILY", RECORDS_PER_CLASS),
            _records("KNOWLEDGE", RECORDS_PER_CLASS),
            _records("CREATIVE", RECORDS_PER_CLASS),
            _records("ANOMALY", RECORDS_PER_CLASS),
        ]

        combined = combine(role_records)

        assert len(combined) == RECORDS_PER_CLASS * 5

    def test_raises_when_a_class_has_wrong_count(self):
        role_records = [
            _records("IT", RECORDS_PER_CLASS - 1),
            _records("DAILY", RECORDS_PER_CLASS),
            _records("KNOWLEDGE", RECORDS_PER_CLASS),
            _records("CREATIVE", RECORDS_PER_CLASS),
            _records("ANOMALY", RECORDS_PER_CLASS),
        ]

        with pytest.raises(DataValidationError, match="IT"):
            combine(role_records)

    def test_raises_when_category_missing(self):
        broken = QueryRecord(query="q", response="r")
        role_records = [
            [broken] + _records("IT", RECORDS_PER_CLASS - 1),
            _records("DAILY", RECORDS_PER_CLASS),
            _records("KNOWLEDGE", RECORDS_PER_CLASS),
            _records("CREATIVE", RECORDS_PER_CLASS),
            _records("ANOMALY", RECORDS_PER_CLASS),
        ]

        with pytest.raises(DataValidationError):
            combine(role_records)

    def test_raises_when_duplicate_query_category_pair(self):
        role_records = [
            _records("IT", RECORDS_PER_CLASS, query_prefix="dup"),
            _records("DAILY", RECORDS_PER_CLASS),
            _records("KNOWLEDGE", RECORDS_PER_CLASS),
            _records("CREATIVE", RECORDS_PER_CLASS),
            _records("ANOMALY", RECORDS_PER_CLASS),
        ]
        # Overwrite one DAILY record's query so it duplicates an IT query+category is impossible
        # (categories differ); instead duplicate within IT itself by reusing the same query text.
        role_records[0][1] = QueryRecord(query="dup0", response="different", category="IT")

        with pytest.raises(DataValidationError, match="중복"):
            combine(role_records)
