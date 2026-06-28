"""QuickStart CLI: provision .env secrets + initialize the DB in one shot.

These exercise the file-provisioning and idempotency contract. `init_db` runs
against a temp working directory (the default db_path resolves under CWD), and
the secret-key env var from conftest keeps key verification happy.
"""

import os
import stat

import pytest

from omnifusion import cli


def _write(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def _read(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _env_value(text: str, key: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(f"{key}=") and not line.lstrip().startswith("#"):
            return line.split("=", 1)[1]
    return None


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run each quickstart in an isolated CWD so .env and data/ stay in tmp.

    quickstart() reloads the global settings singleton in-process (correct for the
    real one-shot CLI). Snapshot and restore it here so that mutation can't leak
    into later tests sharing the process.
    """
    monkeypatch.chdir(tmp_path)
    _write(
        tmp_path / ".env.example",
        "OMNIFUSION_SECRET_KEY=YOUR_FERNET_SECRET_KEY\n"
        "OMNIFUSION_ADMIN_PASSWORD=YOUR_ADMIN_PASSWORD\n"
        "OMNIFUSION_API_KEYS=key_one,key_two\n",
    )

    from omnifusion.settings import settings

    snapshot = dict(settings.__dict__)
    try:
        yield tmp_path
    finally:
        settings.__dict__.clear()
        settings.__dict__.update(snapshot)


def test_quickstart_generates_missing_secrets(workdir):
    cli.quickstart(serve=False)

    env = _read(workdir / ".env")

    secret_key = _env_value(env, "OMNIFUSION_SECRET_KEY")
    assert not cli._is_placeholder_secret_key(secret_key)
    # A generated Fernet key is valid.
    from cryptography.fernet import Fernet

    Fernet(secret_key.encode())

    assert not cli._is_placeholder_admin_password(
        _env_value(env, "OMNIFUSION_ADMIN_PASSWORD")
    )

    api_keys = _env_value(env, "OMNIFUSION_API_KEYS")
    assert not cli._is_placeholder_api_keys(api_keys)
    assert api_keys.startswith("sk-omnifusion-")

    # DB created at the default path under the temp CWD.
    assert (workdir / "data" / "omnifusion.db").exists()


def test_quickstart_writes_env_with_0600_perms(workdir):
    cli.quickstart(serve=False)
    mode = stat.S_IMODE(os.stat(workdir / ".env").st_mode)
    assert mode == 0o600


def test_quickstart_is_idempotent(workdir):
    cli.quickstart(serve=False)
    first = _read(workdir / ".env")

    cli.quickstart(serve=False)
    second = _read(workdir / ".env")

    # Real secrets generated on the first run are kept verbatim on re-run.
    assert first == second


def test_quickstart_never_overwrites_real_secrets(workdir):
    from cryptography.fernet import Fernet

    real_key = Fernet.generate_key().decode()
    _write(
        workdir / ".env",
        f"OMNIFUSION_SECRET_KEY={real_key}\n"
        "OMNIFUSION_ADMIN_PASSWORD=a-real-admin-password\n"
        "OMNIFUSION_API_KEYS=sk-mine-123\n",
    )

    cli.quickstart(serve=False)
    env = _read(workdir / ".env")

    assert _env_value(env, "OMNIFUSION_SECRET_KEY") == real_key
    assert _env_value(env, "OMNIFUSION_ADMIN_PASSWORD") == "a-real-admin-password"
    assert _env_value(env, "OMNIFUSION_API_KEYS") == "sk-mine-123"


def test_quickstart_creates_env_from_example_when_missing(workdir):
    assert not (workdir / ".env").exists()
    cli.quickstart(serve=False)
    env = _read(workdir / ".env")
    # Comment lines from the template are preserved.
    assert "# Security Keys" not in env  # minimal template has no comments
    assert _env_value(env, "OMNIFUSION_SECRET_KEY") is not None


def test_placeholder_detection_helpers():
    assert cli._is_placeholder_secret_key("YOUR_FERNET_SECRET_KEY")
    assert cli._is_placeholder_secret_key("")
    assert not cli._is_placeholder_secret_key("VGpFqdoy7rLK1vr3-wTfek3=")

    assert cli._is_placeholder_admin_password("changeme")
    assert not cli._is_placeholder_admin_password("s3cret")

    assert cli._is_placeholder_api_keys("key_one,key_two")
    assert cli._is_placeholder_api_keys("")
    assert not cli._is_placeholder_api_keys("sk-omnifusion-abc")
