"""Phase 5 예측 파이프라인 오케스트레이션 — P5_설계서_Inference.md 4절. 등급 B
(오케스트레이션) — TextClassifier Protocol 하나에만 의존(DIP), fake로 통합 테스트.
분류 방식(임베딩+LR/NLI/앙상블 등)이 무엇이든 이 모듈은 알지 못한다.

Trigger: predict(classifier, items) — inference/api.py가 감싼다.
Input:   TextClassifier 구현체, list[QueryRecord](분류 요청 항목들)
Output:  list[PredictionResult] — items와 순서로 대응
"""

from embedding_lr.domain.interfaces import TextClassifier
from embedding_lr.domain.models import PredictionResult, QueryRecord
from embedding_lr.evaluation import metrics
from embedding_lr.preprocessing.text_cleaner import clean_text


def predict(classifier: TextClassifier, items: list[QueryRecord]) -> list[PredictionResult]:
    """items가 비어 있으면 classifier.classify() 호출 없이 빈 리스트 반환. item.response는
    읽지 않는다(학습 데이터의 임베딩 대상이 질의 단독이므로 — P5_요구사항정의서_Inference.md
    2절)."""
    if not items:
        return []

    cleaned_queries = [clean_text(item.query) for item in items]
    probs = classifier.classify(cleaned_queries)
    predicted = metrics.probs_to_labels(probs)
    binary = metrics.to_binary_labels(predicted)

    return [
        PredictionResult(predicted_category=category, final_verdict=verdict, probabilities=prob)
        for category, verdict, prob in zip(predicted, binary, probs)
    ]
