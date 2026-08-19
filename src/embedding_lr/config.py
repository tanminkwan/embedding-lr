"""환경 설정 — P0_설계서_Common.md 2절. 환경마다 달라지는 값은 .env에서 읽는다."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    aipro_base_url: str
    aipro_api_token: str
    aipro_timeout_seconds: float = 30.0

    embedding_server_base_url: str
    embedding_server_timeout_seconds: float = 30.0

    log_level: str = "INFO"
    service_name: str = "embedding_lr"
    env: str = "local"

    model_dir: str
    status_dir: str = "status"

    model_config = SettingsConfigDict(env_file=".env", protected_namespaces=())
