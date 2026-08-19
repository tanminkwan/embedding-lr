"""임베딩 입력 텍스트 정제 — P2_설계서_TextCleaning.md. 등급 A(순수 로직) — 테스트 먼저.

순서 고정: 코드펜스 구분자 제거 → 스택 트레이스 라인 제거 → 공백/개행 정규화.
이 순서를 바꾸면 안 된다(P2_설계서_TextCleaning.md 4절)."""

import re

_FENCE_LINE = re.compile(r"^```[a-zA-Z]*$")
_STACK_TRACE_PATTERNS = (
    re.compile(r"[A-Za-z]+Exception\b"),
    re.compile(r"Caused by"),
    re.compile(r"at \S+\("),
    re.compile(r"Traceback"),
)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{2,}")


def clean_text(text: str) -> str:
    text = _strip_fence_lines(text)
    text = _strip_stack_trace_lines(text)
    text = _normalize_whitespace(text)
    return text


def _strip_fence_lines(text: str) -> str:
    lines = [line for line in text.split("\n") if not _FENCE_LINE.match(line.strip())]
    return "\n".join(lines)


def _strip_stack_trace_lines(text: str) -> str:
    lines = [
        line
        for line in text.split("\n")
        if not any(pattern.search(line) for pattern in _STACK_TRACE_PATTERNS)
    ]
    return "\n".join(lines)


def _normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n", text)
    return text.strip()
