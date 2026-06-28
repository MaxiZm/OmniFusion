from fastapi.testclient import TestClient

from omnifusion.settings import settings


def test_missing_auth_uses_openai_error_envelope(tmp_path):
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m2_errors.db")
    settings.omnifusion_api_keys = ["error-key"]

    try:
        with TestClient(app) as client:
            response = client.get("/v1/models")
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["code"] == "unauthorized"


def test_api_http_exception_uses_openai_error_envelope(tmp_path):
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m2_trace_error.db")
    settings.omnifusion_api_keys = ["trace-key"]

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/traces/missing",
                headers={"Authorization": "Bearer trace-key"},
            )
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Trace not found or not stored"
    assert response.json()["error"]["code"] == "not_found"
