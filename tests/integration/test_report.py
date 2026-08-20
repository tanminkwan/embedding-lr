import pytest

from embedding_lr.domain.models import HyperparamSearchResult
from embedding_lr.evaluation import report


class _FakeClassifier:
    """고정 확률을 반환하는 fake Classifier — sklearn 없이 report.build_report 검증."""

    def __init__(self, fixed_probs: list[dict[str, float]]) -> None:
        self._fixed_probs = fixed_probs

    def fit(self, X, y) -> None:  # pragma: no cover - Classifier Protocol 준수용
        raise NotImplementedError

    def predict_proba(self, X: list[list[float]]) -> list[dict[str, float]]:
        return self._fixed_probs


def _all_it_probs(n: int) -> list[dict[str, float]]:
    row = {"IT": 0.9, "DAILY": 0.025, "KNOWLEDGE": 0.025, "CREATIVE": 0.025, "ANOMALY": 0.025}
    return [row] * n


def _search_result(best_accuracy=0.9, best_f1_macro=0.88) -> HyperparamSearchResult:
    return HyperparamSearchResult(
        best_params={"C": 1.0}, best_accuracy=best_accuracy, best_f1_macro=best_f1_macro, trials=[]
    )


class TestBuildReport:
    def test_builds_report_from_model_predictions(self):
        model = _FakeClassifier(_all_it_probs(4))
        X_val = [[0.0]] * 4
        y_val = ["IT", "IT", "DAILY", "DAILY"]

        result = report.build_report(model, X_val, y_val, _search_result(), gap_warning_threshold=0.1)

        assert result.metrics.accuracy == pytest.approx(0.5)
        assert result.targets.accuracy_target_met is False
        assert result.gap.accuracy_gap == pytest.approx(0.4)
        assert result.gap.warning is True


class TestRenderMarkdown:
    def test_includes_key_sections(self):
        model = _FakeClassifier(_all_it_probs(2))
        result = report.build_report(model, [[0.0]] * 2, ["IT", "IT"], _search_result(), gap_warning_threshold=0.1)

        markdown = report.render_markdown(result)

        assert "# Phase 4 검증 결과" in markdown
        assert "Confusion Matrix" in markdown
        assert "Classification Report" in markdown


class TestSaveReport:
    def test_writes_md_and_json(self, tmp_path):
        model = _FakeClassifier(_all_it_probs(2))
        result = report.build_report(model, [[0.0]] * 2, ["IT", "IT"], _search_result(), gap_warning_threshold=0.1)
        md_path = tmp_path / "eval_report.md"
        json_path = tmp_path / "eval_report.json"

        report.save_report(result, str(md_path), str(json_path))

        assert md_path.exists()
        assert json_path.exists()

    def test_raises_and_writes_nothing_when_md_already_exists(self, tmp_path):
        model = _FakeClassifier(_all_it_probs(2))
        result = report.build_report(model, [[0.0]] * 2, ["IT", "IT"], _search_result(), gap_warning_threshold=0.1)
        md_path = tmp_path / "eval_report.md"
        json_path = tmp_path / "eval_report.json"
        md_path.write_text("existing")

        with pytest.raises(Exception, match="이미 존재"):
            report.save_report(result, str(md_path), str(json_path))

        assert not json_path.exists()

    def test_raises_and_writes_nothing_when_json_already_exists(self, tmp_path):
        model = _FakeClassifier(_all_it_probs(2))
        result = report.build_report(model, [[0.0]] * 2, ["IT", "IT"], _search_result(), gap_warning_threshold=0.1)
        md_path = tmp_path / "eval_report.md"
        json_path = tmp_path / "eval_report.json"
        json_path.write_text("{}")

        with pytest.raises(Exception, match="이미 존재"):
            report.save_report(result, str(md_path), str(json_path))

        assert not md_path.exists()
