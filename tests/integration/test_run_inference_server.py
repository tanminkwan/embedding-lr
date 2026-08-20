import pytest
from fastapi import FastAPI

from embedding_lr.cli.run_inference_server import main
from embedding_lr.exceptions import ModelNotFoundError
from embedding_lr.training.persistence import save_model
from embedding_lr.training.trainer import LogisticRegressionClassifier

REQUIRED_ENV = {
    "AIPRO_BASE_URL": "http://localhost:28000",
    "AIPRO_API_TOKEN": "test-token",
    "EMBEDDING_SERVER_BASE_URL": "http://localhost:8000",
    "MODEL_DIR": "models",
}


def _set_env(monkeypatch, tmp_path, model_path):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("STATUS_DIR", str(tmp_path / "status"))
    monkeypatch.setenv("MODEL_PATH", str(model_path))


def _write_fitted_model(path):
    model = LogisticRegressionClassifier(C=1.0, max_iter=500)
    model.fit([[1.0, 0.0], [0.0, 1.0]], ["IT", "DAILY"])
    save_model(model, str(path))


class TestMain:
    def test_starts_uvicorn_with_configured_host_and_port(self, monkeypatch, tmp_path):
        model_path = tmp_path / "model.pkl"
        _write_fitted_model(model_path)
        _set_env(monkeypatch, tmp_path, model_path)
        monkeypatch.setenv("INFERENCE_HOST", "127.0.0.1")
        monkeypatch.setenv("INFERENCE_PORT", "9090")

        captured = {}

        def _fake_run(app, host, port):
            captured["app"] = app
            captured["host"] = host
            captured["port"] = port

        monkeypatch.setattr("embedding_lr.cli.run_inference_server.uvicorn.run", _fake_run)

        main()

        assert isinstance(captured["app"], FastAPI)
        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 9090

    def test_raises_when_model_file_missing(self, monkeypatch, tmp_path):
        missing_model_path = tmp_path / "missing.pkl"
        _set_env(monkeypatch, tmp_path, missing_model_path)

        called = []
        monkeypatch.setattr(
            "embedding_lr.cli.run_inference_server.uvicorn.run",
            lambda *a, **k: called.append(True),
        )

        with pytest.raises(ModelNotFoundError):
            main()

        assert called == []
