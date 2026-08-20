import json

import pandas as pd

from embedding_lr.cli.run_phase3 import main
from embedding_lr.domain.models import HyperparamSearchResult
from embedding_lr.training.persistence import load_model

REQUIRED_ENV = {
    "AIPRO_BASE_URL": "http://localhost:28000",
    "AIPRO_API_TOKEN": "test-token",
    "EMBEDDING_SERVER_BASE_URL": "http://localhost:8000",
    "MODEL_DIR": "models",
}


def _set_env(monkeypatch, tmp_path):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("STATUS_DIR", str(tmp_path / "status"))


def _write_vectors_parquet(path, n_per_class):
    embeddings, labels = [], []
    for i in range(n_per_class):
        embeddings.append([1.0, 0.0, i * 0.01, 0.0])
        labels.append("IT")
        embeddings.append([0.0, 1.0, 0.0, i * 0.01])
        labels.append("DAILY")
    pd.DataFrame({"embedding": embeddings, "label": labels}).to_parquet(path, index=False)


def _write_config(path):
    path.write_text(json.dumps({"C": [0.1, 1.0], "solver": ["lbfgs"], "max_iter": [200]}))


class TestRunPhase3:
    def test_trains_and_saves_model_and_search_result(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        train_path = tmp_path / "train_vectors.parquet"
        test_path = tmp_path / "test_vectors.parquet"
        config_path = tmp_path / "hyperparams.json"
        model_output = tmp_path / "model.pkl"
        search_output = tmp_path / "hyperparams_result.json"
        _write_vectors_parquet(train_path, n_per_class=10)
        _write_vectors_parquet(test_path, n_per_class=5)
        _write_config(config_path)

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_phase3",
                "--train", str(train_path),
                "--test", str(test_path),
                "--model-output", str(model_output),
                "--search-output", str(search_output),
                "--config", str(config_path),
            ],
        )

        main()

        model = load_model(str(model_output))
        probabilities = model.predict_proba([[1.0, 0.0, 0.0, 0.0]])
        assert set(probabilities[0].keys()) == {"IT", "DAILY"}

        result = HyperparamSearchResult.model_validate_json(search_output.read_text(encoding="utf-8"))
        assert len(result.trials) == 2

    def test_records_succeeded_status(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        train_path = tmp_path / "train_vectors.parquet"
        test_path = tmp_path / "test_vectors.parquet"
        config_path = tmp_path / "hyperparams.json"
        model_output = tmp_path / "model.pkl"
        search_output = tmp_path / "hyperparams_result.json"
        _write_vectors_parquet(train_path, n_per_class=10)
        _write_vectors_parquet(test_path, n_per_class=5)
        _write_config(config_path)

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_phase3",
                "--train", str(train_path),
                "--test", str(test_path),
                "--model-output", str(model_output),
                "--search-output", str(search_output),
                "--config", str(config_path),
            ],
        )

        main()

        status_files = list((tmp_path / "status").glob("phase3_*.json"))
        assert len(status_files) == 1
        status = json.loads(status_files[0].read_text(encoding="utf-8"))
        assert status["status"] == "succeeded"
