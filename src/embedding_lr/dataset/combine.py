"""role_*.jsonl에서 로드된 QueryRecord 9묶음 재조합 — P1_설계서_DataPreparation.md 4절.
등급 A(순수 로직) — 파일/포맷을 모르고 list[QueryRecord] 위에서만 동작한다."""

from collections import Counter

from embedding_lr.constants import CLASS_LABELS, RECORDS_PER_CLASS
from embedding_lr.domain.models import QueryRecord
from embedding_lr.exceptions import DataValidationError


def combine(role_records: list[list[QueryRecord]]) -> list[QueryRecord]:
    combined = [record for records in role_records for record in records]

    for record in combined:
        if record.category is None:
            raise DataValidationError(f"카테고리 누락 레코드 존재: {record.query!r}")

    counts = Counter(record.category for record in combined)
    mismatches = {
        label: counts.get(label, 0)
        for label in CLASS_LABELS
        if counts.get(label, 0) != RECORDS_PER_CLASS
    }
    if mismatches:
        detail = ", ".join(
            f"{label} {count}건(기대 {RECORDS_PER_CLASS}건)" for label, count in mismatches.items()
        )
        raise DataValidationError(f"클래스별 건수 불일치: {detail}")

    seen: set[tuple[str, str]] = set()
    duplicates: set[tuple[str, str]] = set()
    for record in combined:
        key = (record.query, record.category)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise DataValidationError(f"중복 레코드 존재 (질의, 카테고리): {sorted(duplicates)}")

    return combined
