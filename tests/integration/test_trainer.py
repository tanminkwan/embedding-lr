import pandas as pd
import pytest

from embedding_lr.domain.models import HyperparamSearchResult
from embedding_lr.exceptions import DataValidationError
from embedding_lr.training import trainer
from embedding_lr.training.trainer import LogisticRegressionClassifier


def _write_vectors_parquet(path, embeddings, labels):
    pd.DataFrame({"embedding": embeddings, "label": labels}).to_parquet(path, index=False)


def _synthetic_split(n_per_class: int, offset: float = 0.0):
    embeddings, labels = [], []
    for i in range(n_per_class):
        embeddings.append([1.0 + offset, 0.0, i * 0.01, 0.0])
        labels.append("IT")
        embeddings.append([0.0, 1.0 + offset, 0.0, i * 0.01])
        labels.append("DAILY")
    return embeddings, labels


class TestLoadVectors:
    def test_returns_embedding_and_label_lists(self, tmp_path):
        path = tmp_path / "train_vectors.parquet"
        embeddings, labels = _synthetic_split(3)
        _write_vectors_parquet(path, embeddings, labels)

        X, y = trainer.load_vectors(str(path))

        assert len(X) == len(y) == 6
        assert X[0] == embeddings[0]
        assert set(y) == {"IT", "DAILY"}

    def test_raises_when_column_missing(self, tmp_path):
        path = tmp_path / "train_vectors.parquet"
        pd.DataFrame({"embedding": [[1.0, 2.0]]}).to_parquet(path, index=False)

        with pytest.raises(DataValidationError, match="필수 컬럼 누락"):
            trainer.load_vectors(str(path))

    def test_raises_when_label_unknown(self, tmp_path):
        path = tmp_path / "train_vectors.parquet"
        _write_vectors_parquet(path, [[1.0, 2.0]], ["NOT_A_LABEL"])

        with pytest.raises(DataValidationError, match="알 수 없는 label"):
            trainer.load_vectors(str(path))


class TestLogisticRegressionClassifier:
    def test_predict_proba_returns_label_keyed_dicts_summing_to_one(self):
        X_train, y_train = _synthetic_split(10)
        classifier = LogisticRegressionClassifier(C=1.0, max_iter=500)

        classifier.fit(X_train, y_train)
        probabilities = classifier.predict_proba([[1.0, 0.0, 0.0, 0.0]])

        assert len(probabilities) == 1
        assert set(probabilities[0].keys()) == {"IT", "DAILY"}
        assert probabilities[0]["IT"] == pytest.approx(sum(probabilities[0].values()) - probabilities[0]["DAILY"])
        assert sum(probabilities[0].values()) == pytest.approx(1.0)


class TestSearchHyperparameters:
    def test_returns_result_covering_every_grid_combination(self):
        X_train, y_train = _synthetic_split(10)
        X_test, y_test = _synthetic_split(5, offset=0.1)
        param_grid = {"C": [0.1, 1.0], "solver": ["lbfgs"], "max_iter": [200]}

        result = trainer.search_hyperparameters(X_train, y_train, X_test, y_test, param_grid)

        assert isinstance(result, HyperparamSearchResult)
        assert len(result.trials) == 2
        assert result.best_params in ({"C": 0.1, "solver": "lbfgs", "max_iter": 200}, {"C": 1.0, "solver": "lbfgs", "max_iter": 200})
        assert 0.0 <= result.best_accuracy <= 1.0
        assert 0.0 <= result.best_f1_macro <= 1.0

    def test_breaks_f1_macro_tie_using_accuracy(self, monkeypatch):
        class _FakeGridSearchCV:
            def __init__(self, estimator, param_grid, scoring, cv, refit) -> None:
                self.cv_results_ = None

            def fit(self, X, y):
                self.cv_results_ = {
                    "params": [
                        {"C": 0.1},
                        {"C": 1.0},
                        {"C": 10.0},
                    ],
                    "mean_test_f1_macro": [0.80, 0.90, 0.90],
                    "mean_test_accuracy": [0.85, 0.70, 0.95],
                }

        monkeypatch.setattr("embedding_lr.training.trainer.GridSearchCV", _FakeGridSearchCV)

        result = trainer.search_hyperparameters([[0.0]], ["IT"], [[0.0]], ["IT"], {"C": [0.1, 1.0, 10.0]})

        assert result.best_params == {"C": 10.0}
        assert result.best_f1_macro == pytest.approx(0.90)
        assert result.best_accuracy == pytest.approx(0.95)
        assert len(result.trials) == 3

    def test_ignores_nan_scores_when_selecting_best(self, monkeypatch):
        """liblinear + 3-class 이상처럼 sklearn이 fit을 거부하는 조합은 NaN 점수로
        기록되는데, 이런 조합이 최적으로 잘못 선정되면 안 된다."""

        class _FakeGridSearchCV:
            def __init__(self, estimator, param_grid, scoring, cv, refit) -> None:
                self.cv_results_ = None

            def fit(self, X, y):
                self.cv_results_ = {
                    "params": [
                        {"C": 0.1, "solver": "liblinear"},
                        {"C": 1.0, "solver": "lbfgs"},
                        {"C": 10.0, "solver": "liblinear"},
                    ],
                    "mean_test_f1_macro": [float("nan"), 0.85, float("nan")],
                    "mean_test_accuracy": [float("nan"), 0.86, float("nan")],
                }

        monkeypatch.setattr("embedding_lr.training.trainer.GridSearchCV", _FakeGridSearchCV)

        result = trainer.search_hyperparameters(
            [[0.0]], ["IT"], [[0.0]], ["IT"], {"C": [0.1, 1.0, 10.0], "solver": ["liblinear", "lbfgs"]}
        )

        assert result.best_params == {"C": 1.0, "solver": "lbfgs"}
        assert result.best_f1_macro == pytest.approx(0.85)
        assert len(result.trials) == 3

    def test_raises_when_every_combination_fails(self, monkeypatch):
        class _FakeGridSearchCV:
            def __init__(self, estimator, param_grid, scoring, cv, refit) -> None:
                self.cv_results_ = None

            def fit(self, X, y):
                self.cv_results_ = {
                    "params": [{"C": 0.1}, {"C": 1.0}],
                    "mean_test_f1_macro": [float("nan"), float("nan")],
                    "mean_test_accuracy": [float("nan"), float("nan")],
                }

        monkeypatch.setattr("embedding_lr.training.trainer.GridSearchCV", _FakeGridSearchCV)

        with pytest.raises(DataValidationError, match="fit에 실패"):
            trainer.search_hyperparameters([[0.0]], ["IT"], [[0.0]], ["IT"], {"C": [0.1, 1.0]})


class TestTrainFinalModel:
    def test_returns_classifier_fitted_only_on_train_set(self):
        X_train, y_train = _synthetic_split(10)

        model = trainer.train_final_model(X_train, y_train, {"C": 1.0, "max_iter": 500})

        assert isinstance(model, LogisticRegressionClassifier)
        probabilities = model.predict_proba([[1.0, 0.0, 0.0, 0.0]])
        assert set(probabilities[0].keys()) == {"IT", "DAILY"}
