"""도메인 상수 — P0_설계서_Common.md 1절. 환경에 따라 달라지지 않는 고정값만 둔다."""

CLASS_LABELS = ["IT", "DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"]
IT_LABEL = "IT"
NON_IT_LABELS = ["DAILY", "KNOWLEDGE", "CREATIVE", "ANOMALY"]

EMBEDDING_DIM = 1024

DATA_SPLITS = ["train", "test", "validation"]

DOMAIN_NAME = "embedding_lr"

CSV_COLUMN_QUERY = "질의"
CSV_COLUMN_RESPONSE = "응답"
CSV_COLUMN_CATEGORY = "카테고리"
