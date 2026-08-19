"""로깅 설정 — P0_설계서_Logging.md. stdout에 구조화된 JSON 로그 출력."""

import logging
import sys
from datetime import datetime, timezone

from pythonjsonlogger import json as jsonlogger

from embedding_lr.config import Settings

_EXTRA_FIELDS = ("phase", "run_id", "extra")


class _JsonFormatter(jsonlogger.JsonFormatter):
    def __init__(self, service: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._service = service

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record.pop("levelname", None)
        log_record.pop("name", None)
        log_record["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        log_record["level"] = record.levelname
        log_record["service"] = self._service
        log_record["logger"] = record.name
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                log_record[field] = value


def setup_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_JsonFormatter(service=settings.service_name))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level)
