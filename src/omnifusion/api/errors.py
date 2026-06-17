from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("omnifusion.errors")


class OmniFusionError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        type_: str = "invalid_request_error",
        code: str = None,
    ):
        self.message = message
        self.status_code = status_code
        self.type = type_
        self.code = code


class InsufficientPanelError(OmniFusionError):
    def __init__(self, message: str = "Insufficient panel success"):
        super().__init__(
            message, status_code=503, type_="server_error", code="insufficient_panel"
        )


class BudgetExceededError(OmniFusionError):
    def __init__(self, message: str = "Budget exceeded"):
        super().__init__(
            message, status_code=402, type_="budget_error", code="budget_exceeded"
        )


class ConfigurationError(OmniFusionError):
    def __init__(self, message: str):
        super().__init__(
            message, status_code=500, type_="server_error", code="configuration_error"
        )


def _run_id_headers(request: Request) -> dict:
    run_id = getattr(request.state, "run_id", None)
    return {"X-OmniFusion-Run-Id": run_id} if run_id else {}


async def omnifusion_error_handler(request: Request, exc: OmniFusionError):
    return JSONResponse(
        status_code=exc.status_code,
        headers=_run_id_headers(request),
        content={
            "error": {
                "message": exc.message,
                "type": exc.type,
                "param": None,
                "code": exc.code,
            }
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        headers=_run_id_headers(request),
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "param": None,
                "code": None,
            }
        },
    )
