"""version+split → AIPro+ 콜렉션명 생성 규칙 — Architecture_Design.md 4절, Scope_Definition.md 2.1절.
등급 A(순수 로직, 외부 의존성 없음) — 테스트 먼저 작성."""

from pathlib import Path

from embedding_lr.constants import DATA_SPLITS, SPLIT_FILE_STEMS
from embedding_lr.exceptions import DataValidationError

_STEM_TO_SPLIT = {stem: split for split, stem in SPLIT_FILE_STEMS.items()}


def collection_name(version: str, split: str) -> str:
    """`<version>_<split>` 콜렉션명 생성. AIPro+ collection_name 패턴(`^[a-zA-Z0-9_-]+$`,
    점 금지, 실제 422 확인됨 — 2026-08-19)에 맞춰 version의 `.`을 `_`로 치환한다."""
    if split not in DATA_SPLITS:
        raise DataValidationError(f"split must be one of {DATA_SPLITS}, got {split!r}")
    sanitized_version = version.replace(".", "_")
    return f"{sanitized_version}_{split}"


def extract_version_and_split(path: str) -> tuple[str, str]:
    """`data/<version>/{train,test,val}.jsonl` 경로에서 version과 split을 추출한다."""
    p = Path(path)
    stem = p.stem
    split = _STEM_TO_SPLIT.get(stem)
    if split is None:
        raise DataValidationError(
            f"알 수 없는 파일명입니다: {path!r} (train/test/val.jsonl만 지원)"
        )
    return p.parent.name, split
