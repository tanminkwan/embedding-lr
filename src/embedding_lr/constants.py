"""도메인 상수 — P0_설계서_Common.md 1절. 환경에 따라 달라지지 않는 고정값만 둔다."""

CLASS_LABELS = ["IT", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"]
IT_LABEL = "IT"
NON_IT_LABELS = ["DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"]

EMBEDDING_DIM = 1024

DATA_SPLITS = ["train", "test", "validation"]
SPLIT_RATIOS = {"train": 3, "test": 1, "validation": 1}
SPLIT_FILE_STEMS = {"train": "train", "test": "test", "validation": "val"}
RANDOM_SEED = 42
RECORDS_PER_CLASS = 200

DOMAIN_NAME = "embedding_lr"

# Phase 4 검증 목표치 — Scope_Definition.md 4.4절, 설계상 고정값(P4_설계서_Validation.md 2절)
TARGET_ACCURACY = 0.85
TARGET_BINARY_ACCURACY = 0.90
TARGET_F1_MACRO = 0.85

FIELD_QUERY = "질의"
FIELD_RESPONSE = "응답"
FIELD_CATEGORY = "카테고리"
