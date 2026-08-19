"""Phase 2 CLI — AIPro+ 지식 데이터 등록 + 임베딩 일괄 조회 → *_vectors.parquet.
Architecture_Design.md 3절 Workflow 규약, embedding/pipeline.py 4절 데이터 흐름.
split(train/test/validation)별로 독립 실행한다.

Trigger: python -m embedding_lr.cli.run_phase2 --input <path> --output <path>
Input:   --input(train/test/val.jsonl 경로 1개, `data/<version>/{train,test,val}.jsonl`)
Output:  --output(`<split>_vectors.parquet` 경로 1개) — 이미 존재하면 실패(입출력 보존 원칙)
"""

import argparse

from embedding_lr.config import Settings
from embedding_lr.data_generation.jsonl_repository import JsonlRepository
from embedding_lr.embedding.aipro_client import AIProClient
from embedding_lr.embedding.pipeline import run as run_pipeline
from embedding_lr.logging_config import setup_logging
from embedding_lr.workflow.run_context import run_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: AIPro+ 지식 데이터 등록 + 임베딩 일괄 조회")
    parser.add_argument("--input", required=True, help="train/test/val.jsonl 경로")
    parser.add_argument("--output", required=True, help="<split>_vectors.parquet 출력 경로")
    args = parser.parse_args()

    settings = Settings()
    setup_logging(settings)

    with run_context("phase2", settings) as (run_id, logger):
        logger.info("Phase 2 시작", extra={"extra": {"input": args.input, "output": args.output}})
        store = AIProClient(settings)
        try:
            run_pipeline(JsonlRepository(), store, args.input, args.output)
        finally:
            store.close()
        logger.info("Phase 2 완료", extra={"extra": {"output": args.output}})


if __name__ == "__main__":
    main()
