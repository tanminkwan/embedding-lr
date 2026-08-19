"""run_id·상태 파일 공통 처리 — P0_설계서_Common.md 6절.

Architecture_Design.md 3절 "모니터링" 규약(status/<phase>_<run_id>.json)과
P0_설계서_Logging.md의 run_id 필드를 한 곳에서 구현한다.
"""

import json
import logging
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from embedding_lr.config import Settings


def new_run_id() -> str:
    """포맷: <YYYYMMDD-HHMMSS>-<4자리 랜덤 hex>"""
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _write_status(path: Path, *, run_id: str, phase: str, started_at: str, status: str,
                   ended_at: str | None = None, error: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "phase": phase,
        "started_at": started_at,
        "ended_at": ended_at,
        "status": status,
        "error": error,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


@contextmanager
def run_context(phase: str, settings: Settings):
    """시작 시 status 파일 기록, 정상 종료 시 succeeded, 예외 시 failed 기록 후 재전파.

    사용: `with run_context("phase2", settings) as (run_id, logger): ...`
    """
    run_id = new_run_id()
    status_path = Path(settings.status_dir) / f"{phase}_{run_id}.json"
    started_at = _now_iso()
    _write_status(status_path, run_id=run_id, phase=phase, started_at=started_at, status="started")

    logger = logging.LoggerAdapter(
        logging.getLogger(f"embedding_lr.{phase}"), {"phase": phase, "run_id": run_id}
    )

    try:
        yield run_id, logger
    except Exception as exc:
        _write_status(
            status_path, run_id=run_id, phase=phase, started_at=started_at,
            status="failed", ended_at=_now_iso(), error=str(exc),
        )
        raise
    else:
        _write_status(
            status_path, run_id=run_id, phase=phase, started_at=started_at,
            status="succeeded", ended_at=_now_iso(),
        )
