import pytest

from embedding_lr.constants import RECORDS_PER_CLASS
from embedding_lr.dataset.split import split
from embedding_lr.domain.models import QueryRecord
from embedding_lr.exceptions import DataValidationError


def _records(category: str, count: int) -> list[QueryRecord]:
    return [
        QueryRecord(query=f"{category}-{i}", response=f"r{i}", category=category)
        for i in range(count)
    ]


def _dataset() -> list[QueryRecord]:
    records = []
    for category in ("IT", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"):
        records.extend(_records(category, RECORDS_PER_CLASS))
    return records


class TestSplit:
    def test_splits_into_3_1_1_ratio_per_class(self):
        result = split(_dataset())

        assert set(result.keys()) == {"train", "test", "validation"}
        assert len(result["train"]) == 120 * 5
        assert len(result["test"]) == 40 * 5
        assert len(result["validation"]) == 40 * 5
        for category in ("IT", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"):
            assert sum(1 for r in result["train"] if r.category == category) == 120
            assert sum(1 for r in result["test"] if r.category == category) == 40
            assert sum(1 for r in result["validation"] if r.category == category) == 40

    def test_no_record_lost_or_duplicated_across_splits(self):
        dataset = _dataset()
        result = split(dataset)

        all_queries = [r.query for r in result["train"] + result["test"] + result["validation"]]
        assert sorted(all_queries) == sorted(r.query for r in dataset)

    def test_is_reproducible_with_same_seed(self):
        dataset = _dataset()

        result_a = split(dataset, seed=7)
        result_b = split(dataset, seed=7)

        assert [r.query for r in result_a["train"]] == [r.query for r in result_b["train"]]
        assert [r.query for r in result_a["test"]] == [r.query for r in result_b["test"]]
        assert [r.query for r in result_a["validation"]] == [r.query for r in result_b["validation"]]

    def test_different_seed_can_produce_different_order(self):
        dataset = _dataset()

        result_a = split(dataset, seed=1)
        result_b = split(dataset, seed=2)

        assert [r.query for r in result_a["train"]] != [r.query for r in result_b["train"]]

    def test_raises_when_class_count_not_divisible_by_ratio_sum(self):
        records = _records("IT", RECORDS_PER_CLASS - 1)

        with pytest.raises(DataValidationError, match="IT"):
            split(records)

    def test_raises_when_category_missing(self):
        records = [QueryRecord(query="q", response="r")]

        with pytest.raises(DataValidationError):
            split(records)
