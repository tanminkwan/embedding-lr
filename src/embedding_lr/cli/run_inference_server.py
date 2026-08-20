"""Phase 5 CLI — 추론 서비스 기동 진입점(합성 루트). Architecture_Design.md 2절/6절,
P5_설계서_Inference.md 7절.

Trigger: python -m embedding_lr.cli.run_inference_server
Input:   환경변수(.env) — MODEL_PATH(필수), INFERENCE_HOST/INFERENCE_PORT(선택),
         EMBEDDING_SERVER_BASE_URL(기존 설정 재사용)
Output:  없음(상시 프로세스) — 모델 로드 실패(ModelNotFoundError) 시 기동 자체가 실패한다.

이 파일이 "임베딩+LR 방식을 쓴다"는 결정이 코드에 등장하는 유일한 곳이다. 다른 분류
방식(NLI 등)이나 앙상블로 교체하려면 model/embedding_client/classifier 조립 부분만
바꾸면 되고, inference/predictor.py, inference/api.py는 무수정이다.
"""

import uvicorn

from embedding_lr.config import Settings
from embedding_lr.embedding.embedding_server_client import EmbeddingServerClient
from embedding_lr.inference.api import create_app
from embedding_lr.inference.embedding_lr_classifier import EmbeddingLRTextClassifier
from embedding_lr.logging_config import setup_logging
from embedding_lr.training import persistence


def main() -> None:
    settings = Settings()
    setup_logging(settings)

    model = persistence.load_model(settings.model_path)
    embedding_client = EmbeddingServerClient(settings)
    classifier = EmbeddingLRTextClassifier(embedding_client, model)
    app = create_app(classifier)

    uvicorn.run(app, host=settings.inference_host, port=settings.inference_port)


if __name__ == "__main__":
    main()
