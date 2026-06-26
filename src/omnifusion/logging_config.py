import json
import logging
import sys
from contextvars import ContextVar
from typing import Optional

_run_id: ContextVar[Optional[str]] = ContextVar("omnifusion_run_id", default=None)


def set_run_id(run_id: Optional[str]) -> None:
    _run_id.set(run_id)


def get_run_id() -> Optional[str]:
    return _run_id.get()


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        run_id = get_run_id()
        if run_id:
            payload["run_id"] = run_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.exc_text:
            payload["exception"] = record.exc_text
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "plain") -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z"))
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    root.addHandler(handler)
