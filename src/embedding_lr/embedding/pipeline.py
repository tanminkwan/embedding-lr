"""Phase 2 임베딩 파이프라인 오케스트레이션 — Architecture_Design.md 2절/4절,
Scope_Definition.md 2.1절. 등급 B(오케스트레이션) — `DataRepository`/`VectorStore`
Protocol에만 의존(DIP), fake로 통합 테스트. `embed()`는 어느 경로에서도 호출하지
않는다 — 벡터는 항상 AIPro+가 계산해 저장한 것을 `get_knowledge()`로 가져온다.

Trigger: run(repo, store, input_path, output_path) — cli/run_phase2.py가 감싼다.
Input:   train/test/val.jsonl 경로 1개 (`data/<version>/{train,test,val}.jsonl`)
Output:  `<split>_vectors.parquet` 경로 1개 — 이미 존재하면 실패(입출력 보존 원칙)
"""

from pathlib import Path

import pandas as pd

from embedding_lr.constants import DOMAIN_NAME
from embedding_lr.domain.interfaces import DataRepository, VectorStore
from embedding_lr.domain.models import KnowledgeItem, QueryRecord
from embedding_lr.embedding.collection import collection_name, extract_version_and_split
from embedding_lr.embedding.knowledge_writer import write_knowledge
from embedding_lr.embedding.registration import ensure_collection, ensure_domain
from embedding_lr.exceptions import DataValidationError
from embedding_lr.preprocessing.text_cleaner import clean_text

_COUNT_CHECK_BUFFER = 1
"""get_knowledge() limit = 입력 레코드 수 + 이 버퍼. 콜렉션에 입력보다 더 많은 건수가
남아있는 이상 상태도 '건수 일치'로 오판하지 않기 위함 — Scope_Definition.md 2.1절."""


def run(repo: DataRepository, store: VectorStore, input_path: str, output_path: str) -> None:
    if Path(output_path).exists():
        raise DataValidationError(f"{output_path} 이미 존재 — 덮어쓰기 금지(입출력 보존 원칙)")

    records = repo.load(input_path)
    version, split = extract_version_and_split(input_path)
    name = collection_name(version, split)

    domain = ensure_domain(store, DOMAIN_NAME)
    ensure_collection(store, name, name)

    limit = len(records) + _COUNT_CHECK_BUFFER
    items = store.get_knowledge(domain.id, name, limit)

    if len(items) != len(records):
        cleaned = [
            QueryRecord(query=clean_text(r.query), response=clean_text(r.response), category=r.category)
            for r in records
        ]
        write_knowledge(store, cleaned, domain.id, name)
        items = store.get_knowledge(domain.id, name, limit)
        if len(items) < len(records):
            raise DataValidationError(
                f"{name}: 재등록 후에도 조회된 건수({len(items)})가 입력 레코드 수"
                f"({len(records)})보다 적습니다"
            )

    _write_vectors_parquet(items, output_path)


def _write_vectors_parquet(items: list[KnowledgeItem], output_path: str) -> None:
    df = pd.DataFrame(
        {
            "embedding": [item.embedding for item in items],
            "label": [item.source for item in items],
        }
    )
    df.to_parquet(output_path, index=False)
