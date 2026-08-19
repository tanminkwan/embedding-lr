import pytest

from embedding_lr.embedding.collection import collection_name, extract_version_and_split
from embedding_lr.exceptions import DataValidationError


class TestCollectionName:
    def test_combines_version_and_split(self):
        assert collection_name("v0.2", "train") == "v0_2_train"

    def test_replaces_every_dot_in_version(self):
        assert collection_name("v0.1.2", "test") == "v0_1_2_test"

    def test_leaves_dotless_version_unchanged(self):
        assert collection_name("v1", "validation") == "v1_validation"

    def test_rejects_unknown_split(self):
        with pytest.raises(DataValidationError, match="split"):
            collection_name("v0.2", "unknown")


class TestExtractVersionAndSplit:
    def test_extracts_version_and_train_split(self):
        assert extract_version_and_split("data/v0.2/train.jsonl") == ("v0.2", "train")

    def test_extracts_version_and_test_split(self):
        assert extract_version_and_split("data/v0.2/test.jsonl") == ("v0.2", "test")

    def test_extracts_version_and_validation_split_from_val_stem(self):
        assert extract_version_and_split("data/v0.2/val.jsonl") == ("v0.2", "validation")

    def test_rejects_unknown_file_stem(self):
        with pytest.raises(DataValidationError, match="data.jsonl"):
            extract_version_and_split("data/v0.2/data.jsonl")
