"""Phase 1 CLI — 레거시 CSV role 파일을 JSONL로 변환. P1_설계서_DataPreparation.md 2절.

이미 확보된 원본 데이터(현재 CSV)를 학습 파이프라인이 쓰는 JSONL로 변환한다 —
새 데이터를 만드는 단계가 아니다.

Trigger: python -m embedding_lr.cli.run_phase1 --input <csv path> --output <jsonl path>
Input:   --input(레거시 CSV role 파일, 예: role_03_network.csv)
Output:  --output(변환된 JSONL 파일) — 이미 존재하면 실패(입출력 보존 원칙)
"""

import argparse

from embedding_lr.config import Settings
from embedding_lr.data_generation.csv_repository import CsvRepository
from embedding_lr.data_generation.jsonl_repository import JsonlRepository
from embedding_lr.logging_config import setup_logging
from embedding_lr.workflow.run_context import run_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: 레거시 CSV role 파일 → JSONL 변환")
    parser.add_argument("--input", required=True, help="변환할 CSV 파일 경로")
    parser.add_argument("--output", required=True, help="출력 JSONL 파일 경로")
    args = parser.parse_args()

    settings = Settings()
    setup_logging(settings)

    with run_context("phase1", settings) as (run_id, logger):
        logger.info("CSV → JSONL 변환 시작", extra={"extra": {"input": args.input}})
        records = CsvRepository().load(args.input)
        JsonlRepository().save(records, args.output)
        logger.info(
            "CSV → JSONL 변환 완료",
            extra={"extra": {"count": len(records), "output": args.output}},
        )


if __name__ == "__main__":
    main()
