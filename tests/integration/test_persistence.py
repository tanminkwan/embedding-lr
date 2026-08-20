import pytest

from embedding_lr.domain.models import HyperparamSearchResult, HyperparamTrial
from embedding_lr.exceptions import DataValidationError, ModelNotFoundError
from embedding_lr.training import persistence
from embedding_lr.training.trainer import LogisticRegressionClassifier


def _fitted_model():
    model = LogisticRegressionClassifier(C=1.0, max_iter=500)
    model.fit([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [0.1, 1.0]], ["IT", "DAILY", "IT", "DAILY"])
    return model


def _search_result():
    return HyperparamSearchResult(
        best_params={"C": 1.0},
        best_accuracy=0.9,
        best_f1_macro=0.88,
        trials=[HyperparamTrial(params={"C": 1.0}, accuracy=0.9, f1_macro=0.88)],
    )


class TestSaveLoadModel:
    def test_round_trips_a_fitted_model(self, tmp_path):
        path = tmp_path / "model.pkl"
        model = _fitted_model()

        persistence.save_model(model, str(path))
        loaded = persistence.load_model(str(path))

        probabilities = loaded.predict_proba([[1.0, 0.0]])
        assert set(probabilities[0].keys()) == {"IT", "DAILY"}

    def test_save_raises_when_path_already_exists(self, tmp_path):
        path = tmp_path / "model.pkl"
        path.write_text("existing")

        with pytest.raises(DataValidationError, match="이미 존재"):
            persistence.save_model(_fitted_model(), str(path))

    def test_load_raises_when_path_missing(self, tmp_path):
        path = tmp_path / "missing.pkl"

        with pytest.raises(ModelNotFoundError):
            persistence.load_model(str(path))


class TestSaveSearchResult:
    def test_writes_json_matching_result(self, tmp_path):
        path = tmp_path / "hyperparams.json"
        result = _search_result()

        persistence.save_search_result(result, str(path))

        assert HyperparamSearchResult.model_validate_json(path.read_text(encoding="utf-8")) == result

    def test_raises_when_path_already_exists(self, tmp_path):
        path = tmp_path / "hyperparams.json"
        path.write_text("{}")

        with pytest.raises(DataValidationError, match="이미 존재"):
            persistence.save_search_result(_search_result(), str(path))


class TestLoadSearchResult:
    def test_round_trips_a_saved_result(self, tmp_path):
        path = tmp_path / "hyperparams.json"
        result = _search_result()
        persistence.save_search_result(result, str(path))

        loaded = persistence.load_search_result(str(path))

        assert loaded == result

    def test_raises_when_path_missing(self, tmp_path):
        path = tmp_path / "missing.json"

        with pytest.raises(ModelNotFoundError):
            persistence.load_search_result(str(path))
