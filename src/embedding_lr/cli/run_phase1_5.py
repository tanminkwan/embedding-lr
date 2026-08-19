"""Phase 1.5 CLI — role_*.jsonl 재조합 + 클래스별 3:1:1 분할. P1_설계서_DataPreparation.md 4/5절.

Trigger: python -m embedding_lr.cli.run_phase1_5 --input-dir <path> --output-dir <path>
Input:   --input-dir(role_01~09_*.jsonl 9개가 있는 디렉터리)
Output:  --output-dir(data.jsonl, train.jsonl, test.jsonl, val.jsonl을 쓸 디렉터리)
         — 각 파일이 이미 존재하면 실패(입출력 보존 원칙)
"""

import argparse
from pathlib import Path

from embedding_lr.config import Settings
from embedding_lr.constants import DATA_SPLITS, SPLIT_FILE_STEMS
from embedding_lr.data_generation.jsonl_repository import JsonlRepository
from embedding_lr.dataset.combine import combine
from embedding_lr.dataset.split import split
from embedding_lr.logging_config import setup_logging
from embedding_lr.workflow.run_context import run_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.5: role_*.jsonl 재조합 + 분할")
    parser.add_argument("--input-dir", required=True, help="role_01~09_*.jsonl이 있는 디렉터리")
    parser.add_argument("--output-dir", required=True, help="data/train/test/val.jsonl 출력 디렉터리")
    args = parser.parse_args()

    settings = Settings()
    setup_logging(settings)

    with run_context("phase1_5", settings) as (run_id, logger):
        repo = JsonlRepository()
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        role_paths = sorted(input_dir.glob("role_*.jsonl"))
        logger.info("role 파일 로드 시작", extra={"extra": {"count": len(role_paths)}})
        role_records = [repo.load(str(p)) for p in role_paths]

        combined = combine(role_records)
        data_path = output_dir / "data.jsonl"
        repo.save(combined, str(data_path))
        logger.info(
            "재조합 완료", extra={"extra": {"count": len(combined), "output": str(data_path)}}
        )

        splits = split(combined)
        for split_name in DATA_SPLITS:
            split_path = output_dir / f"{SPLIT_FILE_STEMS[split_name]}.jsonl"
            repo.save(splits[split_name], str(split_path))
            logger.info(
                "분할 저장 완료",
                extra={
                    "extra": {
                        "split": split_name,
                        "count": len(splits[split_name]),
                        "output": str(split_path),
                    }
                },
            )


if __name__ == "__main__":
    main()
