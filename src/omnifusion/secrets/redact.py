import re
import logging
from typing import Set
from ..settings import settings

# Regex patterns to catch common API key / token formats.
# Fix (medium): Broadened to cover Google AIza*, Azure hex, sk-_ variant,
# and URL userinfo credentials (http://user:password@host).
REDACT_PATTERNS = [
    # OpenAI / Anthropic: sk-... and sk-_... variants
    re.compile(r"(sk-[a-zA-Z0-9_\-]{20,})"),
    # Google API keys (AIza...)
    re.compile(r"(AIza[a-zA-Z0-9_\-]{35,})"),
    # Azure / hex API keys (32+ hex chars)
    re.compile(r"\b([0-9a-fA-F]{32,})\b"),
    # Bearer tokens in Authorization headers
    re.compile(r"bearer\s+([a-zA-Z0-9\-\._~+/]+=*)", re.IGNORECASE),
    # api_key= query-string / form parameter
    re.compile(r"api_key=([^\&\s'\"]+)", re.IGNORECASE),
    re.compile(r"api-key=([^\&\s'\"]+)", re.IGNORECASE),
    # URL userinfo credentials (http://user:password@host)
    re.compile(r"(https?://[^:@\s]+:[^@\s]+@)", re.IGNORECASE),
]


class SecretRedactingFilter(logging.Filter):
    def __init__(self, name: str = ""):
        super().__init__(name)
        self.known_secrets: Set[str] = set()

    def add_secret(self, secret: str):
        if secret and len(secret) > 4:
            self.known_secrets.add(secret)

    def redact(self, msg: str) -> str:
        if not isinstance(msg, str):
            return msg

        # 1. Redact general patterns
        for pattern in REDACT_PATTERNS:
            msg = pattern.sub("[REDACTED]", msg)

        # 2. Redact specifically loaded/configured secrets
        # Load API keys from settings if available
        if settings.omnifusion_api_keys:
            for key in settings.omnifusion_api_keys:
                if len(key) > 4:
                    msg = msg.replace(key, "[REDACTED]")

        # Load admin password if set
        if settings.omnifusion_admin_password:
            pwd = settings.omnifusion_admin_password.get_secret_value()
            if len(pwd) > 4:
                msg = msg.replace(pwd, "[REDACTED]")

        # Also dynamic secrets added during runtime (decrypted provider API keys)
        for secret in self.known_secrets:
            msg = msg.replace(secret, "[REDACTED]")

        return msg

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.redact(record.msg)
        elif record.msg is not None:
            # Non-string msg (e.g. logger.error(exc) or a library logging an object):
            # render and redact it, clearing args so the rendered form is used as-is.
            # Without this, secrets embedded in exception/object reprs would leak.
            record.msg = self.redact(str(record.msg))
            record.args = ()
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self.redact(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            else:
                new_args = []
                for arg in record.args:
                    if isinstance(arg, str):
                        new_args.append(self.redact(arg))
                    else:
                        new_args.append(arg)
                record.args = tuple(new_args)
        # Also redact formatted exc_text if present
        if record.exc_text and isinstance(record.exc_text, str):
            record.exc_text = self.redact(record.exc_text)
        return True


# Global redactor instance
redactor = SecretRedactingFilter()


def setup_logging_redaction():
    """
    Fix #6: Attach the redaction filter to HANDLERS (not loggers).

    Per CPython semantics, filters on logger objects only run for records that
    originate at that logger. Records from child loggers (e.g. litellm, httpx)
    propagate to the root handler *without* re-running ancestor-logger filters.

    By attaching to every existing handler on the root logger (and to loggers
    that may add handlers later), we ensure ALL log records — regardless of
    origin — pass through redaction before being emitted.
    """
    root_logger = logging.getLogger()

    # Attach to all current root handlers
    for handler in root_logger.handlers:
        if redactor not in handler.filters:
            handler.addFilter(redactor)

    # Install a hook so any future handlers added to the root logger also get the filter.
    # We do this by subclassing the root logger's addHandler method.
    original_add_handler = root_logger.__class__.addHandler

    def _patched_add_handler(self, handler):
        original_add_handler(self, handler)
        if redactor not in handler.filters:
            handler.addFilter(redactor)

    root_logger.__class__.addHandler = _patched_add_handler

    # Also attach directly to well-known loggers' handlers as a belt-and-suspenders measure
    for logger_name in (
        "omnifusion",
        "omnifusion.llm",
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "litellm",
        "httpx",
        "httpcore",
    ):
        log = logging.getLogger(logger_name)
        # Also add as a logger-level filter for loggers that don't propagate
        if redactor not in log.filters:
            log.addFilter(redactor)
        for handler in log.handlers:
            if redactor not in handler.filters:
                handler.addFilter(redactor)
