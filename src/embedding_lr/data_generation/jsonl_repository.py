"""domain.interfaces.DataRepository 구현체(JSONL) — P1_설계서_DataPreparation.md 3절.
등급 B(오케스트레이션, 파일 I/O 있음) — 구현 후 통합 테스트."""

import json
import os

from pydantic import ValidationError as PydanticValidationError

from embedding_lr.constants import FIELD_CATEGORY, FIELD_QUERY, FIELD_RESPONSE
from embedding_lr.domain.models import QueryRecord
from embedding_lr.exceptions import DataValidationError


class JsonlRepository:
    """현재 학습 파이프라인의 저장 형식(JSONL)을 담당하는 DataRepository 구현체.
    형식이 바뀌면 이 클래스만 교체하면 된다 — dataset.combine/split은 무관."""

    def load(self, path: str) -> list[QueryRecord]:
        records: list[QueryRecord] = []
        with open(path, encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataValidationError(f"{path}:{line_number} JSON 파싱 실패: {exc}") from exc
                try:
                    records.append(
                        QueryRecord(
                            query=raw[FIELD_QUERY],
                            response=raw[FIELD_RESPONSE],
                            category=raw.get(FIELD_CATEGORY),
                        )
                    )
                except KeyError as exc:
                    raise DataValidationError(f"{path}:{line_number} 필수 키 누락: {exc}") from exc
                except PydanticValidationError as exc:
                    raise DataValidationError(f"{path}:{line_number} 필드 검증 실패: {exc}") from exc
        return records

    def save(self, records: list[QueryRecord], path: str) -> None:
        if os.path.exists(path):
            raise DataValidationError(f"{path} 이미 존재 — 덮어쓰기 금지(입출력 보존 원칙)")
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                line = json.dumps(
                    {
                        FIELD_QUERY: record.query,
                        FIELD_RESPONSE: record.response,
                        FIELD_CATEGORY: record.category,
                    },
                    ensure_ascii=False,
                )
                f.write(line + "\n")
