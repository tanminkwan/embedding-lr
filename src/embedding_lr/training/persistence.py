"""학습 산출물(모델/탐색 이력) 저장·로드 — P3_설계서_Training.md 5절. 등급 B.

Trigger: save_model(...) / load_model(...) / save_search_result(...) — cli/run_phase3.py,
Phase 5(inference/predictor.py, 미구현)가 감싼다.
Input:   LogisticRegressionClassifier 또는 HyperparamSearchResult, 저장/로드 경로
Output:  .pkl(joblib) / .json — 저장 시 경로가 이미 존재하면 실패(입출력 보존 원칙)
"""

from pathlib import Path

import joblib

from embedding_lr.domain.models import HyperparamSearchResult
from embedding_lr.exceptions import DataValidationError, ModelNotFoundError
from embedding_lr.training.trainer import LogisticRegressionClassifier


def save_model(model: LogisticRegressionClassifier, path: str) -> None:
    """joblib.dump(model, path). path가 이미 존재하면 DataValidationError(덮어쓰기 금지)."""
    if Path(path).exists():
        raise DataValidationError(f"{path} 이미 존재 — 덮어쓰기 금지(입출력 보존 원칙)")
    joblib.dump(model, path)


def load_model(path: str) -> LogisticRegressionClassifier:
    """joblib.load(path). 파일이 없으면 ModelNotFoundError."""
    if not Path(path).exists():
        raise ModelNotFoundError(f"{path} 모델 파일을 찾을 수 없습니다")
    return joblib.load(path)


def save_search_result(result: HyperparamSearchResult, path: str) -> None:
    """result.model_dump_json(indent=2)를 path에 저장. path가 이미 존재하면 DataValidationError."""
    if Path(path).exists():
        raise DataValidationError(f"{path} 이미 존재 — 덮어쓰기 금지(입출력 보존 원칙)")
    Path(path).write_text(result.model_dump_json(indent=2), encoding="utf-8")
