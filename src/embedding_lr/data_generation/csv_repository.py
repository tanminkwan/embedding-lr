"""레거시 CSV 입력 어댑터 — P1_설계서_DataPreparation.md 3절.
DataRepository Protocol의 입력 전용 구현체. 이미 확보된 원본 데이터가 지금은 CSV
형태로 오지만, 형식이 바뀌어도 dataset.combine/split은 이 클래스만 교체하면 된다.
등급 B(오케스트레이션, 파일 I/O 있음) — 구현 후 통합 테스트."""

import csv

from pydantic import ValidationError as PydanticValidationError

from embedding_lr.constants import FIELD_CATEGORY, FIELD_QUERY, FIELD_RESPONSE
from embedding_lr.domain.models import QueryRecord
from embedding_lr.exceptions import DataValidationError


class CsvRepository:
    """DataRepository 구현체 — 레거시 CSV 원본 읽기 전용.
    CSV로의 저장은 지원하지 않는다(이 프로젝트는 JSONL로만 내보낸다 — JsonlRepository)."""

    def load(self, path: str) -> list[QueryRecord]:
        records: list[QueryRecord] = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for line_number, row in enumerate(reader, start=2):  # 1행 = 헤더
                try:
                    records.append(
                        QueryRecord(
                            query=row[FIELD_QUERY],
                            response=row[FIELD_RESPONSE],
                            category=row.get(FIELD_CATEGORY),
                        )
                    )
                except KeyError as exc:
                    raise DataValidationError(f"{path}:{line_number} 필수 컬럼 누락: {exc}") from exc
                except PydanticValidationError as exc:
                    raise DataValidationError(f"{path}:{line_number} 필드 검증 실패: {exc}") from exc
        return records

    def save(self, records: list[QueryRecord], path: str) -> None:
        raise NotImplementedError(
            "CSV 저장은 지원하지 않습니다 — 이 프로젝트는 JSONL만 출력합니다(JsonlRepository 사용)."
        )
