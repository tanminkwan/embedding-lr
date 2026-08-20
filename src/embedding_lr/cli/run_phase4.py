"""Phase 4 CLI — val_vectors.parquet + model.pkl + hyperparams.json → 검증 리포트.
Architecture_Design.md 3절 Workflow 규약, P4_설계서_Validation.md 8절 데이터 흐름.

Trigger: python -m embedding_lr.cli.run_phase4 \
           --val <path> --model <path> --search-result <path> \
           --report-md <path> --report-json <path> \
           [--config <path, 기본값 config/eval_thresholds_default.json>]
Input:   --val(val_vectors.parquet), --model(model_<ver>.pkl),
         --search-result(hyperparams_<ver>.json), --config(선택)
Output:  --report-md(.md), --report-json(.json) — 둘 다 이미 존재하면 실패(입출력 보존 원칙)
Exit code: 항상 0 — 목표 미달 여부는 리포트 내 targets 필드로만 표시된다(루프백은 사람이 판단).
"""

import argparse
import json

from embedding_lr.config import Settings
from embedding_lr.evaluation import report
from embedding_lr.logging_config import setup_logging
from embedding_lr.training import persistence, trainer
from embedding_lr.workflow.run_context import run_context

_DEFAULT_CONFIG_PATH = "config/eval_thresholds_default.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: val set 기준 모델 검증 리포트 생성")
    parser.add_argument("--val", required=True, help="val_vectors.parquet 경로")
    parser.add_argument("--model", required=True, help="검증 대상 모델 .pkl 경로")
    parser.add_argument("--search-result", required=True, help="Phase 3 하이퍼파라미터 탐색 결과 .json 경로")
    parser.add_argument("--report-md", required=True, help="검증 리포트 .md 출력 경로")
    parser.add_argument("--report-json", required=True, help="검증 리포트 .json 출력 경로")
    parser.add_argument("--config", default=_DEFAULT_CONFIG_PATH, help="gap 경고 임계값 JSON 경로")
    args = parser.parse_args()

    settings = Settings()
    setup_logging(settings)

    with run_context("phase4", settings) as (run_id, logger):
        logger.info(
            "Phase 4 시작",
            extra={"extra": {"val": args.val, "model": args.model, "search_result": args.search_result}},
        )

        X_val, y_val = trainer.load_vectors(args.val)
        model = persistence.load_model(args.model)
        search_result = persistence.load_search_result(args.search_result)
        with open(args.config, encoding="utf-8") as f:
            thresholds = json.load(f)

        evaluation_report = report.build_report(
            model, X_val, y_val, search_result, thresholds["gap_warning_threshold"]
        )
        report.save_report(evaluation_report, args.report_md, args.report_json)

        logger.info(
            "Phase 4 완료",
            extra={"extra": {"targets": evaluation_report.targets.model_dump(), "gap": evaluation_report.gap.model_dump()}},
        )


if __name__ == "__main__":
    main()
