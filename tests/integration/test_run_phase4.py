import json

import pandas as pd

from embedding_lr.cli.run_phase4 import main
from embedding_lr.domain.models import EvaluationReport
from embedding_lr.training.persistence import save_model, save_search_result
from embedding_lr.training.trainer import LogisticRegressionClassifier

REQUIRED_ENV = {
    "AIPRO_BASE_URL": "http://localhost:28000",
    "AIPRO_API_TOKEN": "test-token",
    "EMBEDDING_SERVER_BASE_URL": "http://localhost:8000",
    "MODEL_DIR": "models",
    "MODEL_PATH": "models/model.pkl",
}


def _set_env(monkeypatch, tmp_path):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("STATUS_DIR", str(tmp_path / "status"))


def _write_val_vectors(path, n_per_class):
    embeddings, labels = [], []
    for i in range(n_per_class):
        embeddings.append([1.0, 0.0, i * 0.01, 0.0])
        labels.append("IT")
        embeddings.append([0.0, 1.0, 0.0, i * 0.01])
        labels.append("DAILY")
    pd.DataFrame({"embedding": embeddings, "label": labels}).to_parquet(path, index=False)


def _write_fitted_model(path):
    model = LogisticRegressionClassifier(C=1.0, max_iter=500)
    model.fit([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], ["IT", "DAILY"])
    save_model(model, str(path))


def _write_search_result(path):
    from embedding_lr.domain.models import HyperparamSearchResult

    save_search_result(
        HyperparamSearchResult(best_params={"C": 1.0}, best_accuracy=0.9, best_f1_macro=0.88, trials=[]),
        str(path),
    )


def _write_config(path):
    path.write_text(json.dumps({"gap_warning_threshold": 0.1}))


class TestRunPhase4:
    def test_builds_and_saves_report(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        val_path = tmp_path / "val_vectors.parquet"
        model_path = tmp_path / "model.pkl"
        search_result_path = tmp_path / "hyperparams.json"
        config_path = tmp_path / "eval_thresholds.json"
        report_md = tmp_path / "eval_report.md"
        report_json = tmp_path / "eval_report.json"
        _write_val_vectors(val_path, n_per_class=5)
        _write_fitted_model(model_path)
        _write_search_result(search_result_path)
        _write_config(config_path)

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_phase4",
                "--val", str(val_path),
                "--model", str(model_path),
                "--search-result", str(search_result_path),
                "--report-md", str(report_md),
                "--report-json", str(report_json),
                "--config", str(config_path),
            ],
        )

        main()

        assert report_md.exists()
        report = EvaluationReport.model_validate_json(report_json.read_text(encoding="utf-8"))
        assert 0.0 <= report.metrics.accuracy <= 1.0

    def test_records_succeeded_status_even_when_targets_missed(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        val_path = tmp_path / "val_vectors.parquet"
        model_path = tmp_path / "model.pkl"
        search_result_path = tmp_path / "hyperparams.json"
        config_path = tmp_path / "eval_thresholds.json"
        report_md = tmp_path / "eval_report.md"
        report_json = tmp_path / "eval_report.json"
        _write_val_vectors(val_path, n_per_class=5)
        _write_fitted_model(model_path)
        _write_search_result(search_result_path)
        _write_config(config_path)

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_phase4",
                "--val", str(val_path),
                "--model", str(model_path),
                "--search-result", str(search_result_path),
                "--report-md", str(report_md),
                "--report-json", str(report_json),
                "--config", str(config_path),
            ],
        )

        main()

        status_files = list((tmp_path / "status").glob("phase4_*.json"))
        assert len(status_files) == 1
        status = json.loads(status_files[0].read_text(encoding="utf-8"))
        assert status["status"] == "succeeded"
