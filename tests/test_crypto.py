import pytest
import os
from pydantic import SecretStr
from omnifusion.secrets.crypto import (
    encrypt_key,
    decrypt_key,
    generate_key,
    check_secret_key,
)
from omnifusion.settings import settings
from omnifusion.api.errors import ConfigurationError
from omnifusion.store.db import init_db


@pytest.fixture(autouse=True)
def setup_db():
    old_db = settings.db_path
    settings.db_path = "test_crypto.db"
    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)
    yield
    if os.path.exists(settings.db_path):
        try:
            os.remove(settings.db_path)
        except Exception:
            pass
    settings.db_path = old_db


@pytest.mark.asyncio
async def test_crypto_key_lifecycle():
    new_key = generate_key()
    assert len(new_key) > 0

    settings.omnifusion_secret_key = SecretStr(new_key)

    plain = "sk-test-api-key"
    enc = encrypt_key(plain)
    assert enc != plain

    dec = decrypt_key(enc)
    assert dec == plain


@pytest.mark.asyncio
async def test_key_verification_fails_on_wrong_key():
    key1 = generate_key()
    key2 = generate_key()

    settings.omnifusion_secret_key = SecretStr(key1)
    await init_db()

    await check_secret_key()

    settings.omnifusion_secret_key = SecretStr(key2)

    with pytest.raises(ConfigurationError):
        await check_secret_key()
