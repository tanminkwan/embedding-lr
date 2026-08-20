"""Phase 5 FastAPI 앱 — P5_설계서_Inference.md 5절. 등급 B — TextClassifier Protocol
하나에만 의존, FastAPI TestClient + fake로 통합 테스트.

Trigger: create_app(classifier) — cli/run_inference_server.py가 감싸 uvicorn으로 구동.
Input:   TextClassifier 구현체(합성 루트에서 주입)
Output:  FastAPI 앱 — POST /classify, GET /health
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from embedding_lr.domain.interfaces import TextClassifier
from embedding_lr.domain.models import ClassifyRequest, ClassifyResponse
from embedding_lr.exceptions import EmbeddingServerError
from embedding_lr.inference import predictor


def create_app(classifier: TextClassifier) -> FastAPI:
    """FastAPI 앱을 조립해 반환하는 팩토리 함수 — 전역 상태 없이 classifier를 클로저로
    캡처한다. classifier가 무엇으로 구현됐는지(임베딩+LR/NLI/앙상블)는 이 함수의 관심사가
    아니다."""
    app = FastAPI()

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(request: ClassifyRequest) -> ClassifyResponse:
        results = predictor.predict(classifier, request.items)
        return ClassifyResponse(results=results)

    @app.exception_handler(EmbeddingServerError)
    def handle_embedding_server_error(request: Request, exc: EmbeddingServerError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
