"""QueryRecord 목록을 클래스별 3:1:1 stratified 분할 — P1_설계서_DataPreparation.md 5절.
등급 A(순수 로직) — 파일/포맷을 모르고 list[QueryRecord] 위에서만 동작한다."""

import random
from collections import defaultdict

from embedding_lr.constants import RANDOM_SEED, SPLIT_RATIOS
from embedding_lr.domain.models import QueryRecord
from embedding_lr.exceptions import DataValidationError


def split(
    records: list[QueryRecord],
    seed: int = RANDOM_SEED,
    ratios: dict[str, int] = SPLIT_RATIOS,
) -> dict[str, list[QueryRecord]]:
    groups: dict[str, list[QueryRecord]] = defaultdict(list)
    for record in records:
        if record.category is None:
            raise DataValidationError(f"카테고리 누락 레코드 존재: {record.query!r}")
        groups[record.category].append(record)

    ratio_total = sum(ratios.values())
    result: dict[str, list[QueryRecord]] = {name: [] for name in ratios}
    rng = random.Random(seed)

    for category in sorted(groups):
        group = list(groups[category])
        if len(group) % ratio_total != 0:
            raise DataValidationError(
                f"{category} 클래스 건수({len(group)})가 분할 비율 합({ratio_total})으로 "
                "나누어떨어지지 않습니다"
            )
        rng.shuffle(group)
        unit = len(group) // ratio_total
        offset = 0
        for name, ratio in ratios.items():
            count = unit * ratio
            result[name].extend(group[offset : offset + count])
            offset += count

    return result
