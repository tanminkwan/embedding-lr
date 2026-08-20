"""Phase 3 CLI — train/test_vectors.parquet + 하이퍼파라미터 설정 → 최적 LR 모델 학습.
Architecture_Design.md 3절 Workflow 규약, training/trainer.py 4절 데이터 흐름.

Trigger: python -m embedding_lr.cli.run_phase3 \
           --train <path> --test <path> \
           --model-output <path> --search-output <path> \
           [--config <path, 기본값 config/hyperparams_default.json>]
Input:   --train(train_vectors.parquet), --test(test_vectors.parquet), --config(선택)
Output:  --model-output(.pkl), --search-output(.json) — 둘 다 이미 존재하면 실패(입출력 보존 원칙)
"""

import argparse
import json

from embedding_lr.config import Settings
from embedding_lr.logging_config import setup_logging
from embedding_lr.training import persistence, trainer
from embedding_lr.workflow.run_context import run_context

_DEFAULT_CONFIG_PATH = "config/hyperparams_default.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: 하이퍼파라미터 탐색 + 최적 LR 모델 학습")
    parser.add_argument("--train", required=True, help="train_vectors.parquet 경로")
    parser.add_argument("--test", required=True, help="test_vectors.parquet 경로")
    parser.add_argument("--model-output", required=True, help="학습된 모델 .pkl 출력 경로")
    parser.add_argument("--search-output", required=True, help="하이퍼파라미터 탐색 이력 .json 출력 경로")
    parser.add_argument("--config", default=_DEFAULT_CONFIG_PATH, help="하이퍼파라미터 탐색 범위 JSON 경로")
    args = parser.parse_args()

    settings = Settings()
    setup_logging(settings)

    with run_context("phase3", settings) as (run_id, logger):
        logger.info(
            "Phase 3 시작",
            extra={"extra": {"train": args.train, "test": args.test, "config": args.config}},
        )

        X_train, y_train = trainer.load_vectors(args.train)
        X_test, y_test = trainer.load_vectors(args.test)
        with open(args.config, encoding="utf-8") as f:
            param_grid = json.load(f)

        search_result = trainer.search_hyperparameters(X_train, y_train, X_test, y_test, param_grid)
        model = trainer.train_final_model(X_train, y_train, search_result.best_params)

        persistence.save_model(model, args.model_output)
        persistence.save_search_result(search_result, args.search_output)

        logger.info(
            "Phase 3 완료",
            extra={"extra": {"model_output": args.model_output, "best_params": search_result.best_params}},
        )


if __name__ == "__main__":
    main()
