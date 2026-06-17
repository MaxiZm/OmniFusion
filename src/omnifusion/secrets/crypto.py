from cryptography.fernet import Fernet
from ..settings import settings
from ..api.errors import ConfigurationError


def get_fernet() -> Fernet:
    key_str = settings.omnifusion_secret_key
    if not key_str:
        raise ConfigurationError("OMNIFUSION_SECRET_KEY is not set.")
    try:
        return Fernet(key_str.get_secret_value().encode())
    except Exception as e:
        raise ConfigurationError(f"Invalid OMNIFUSION_SECRET_KEY: {e}")


def encrypt_key(plain_key: str) -> bytes:
    if not plain_key:
        return b""
    f = get_fernet()
    return f.encrypt(plain_key.encode())


def decrypt_key(enc_key: bytes) -> str:
    if not enc_key:
        return ""
    f = get_fernet()
    return f.decrypt(enc_key).decode()


def generate_key() -> str:
    return Fernet.generate_key().decode()


async def _check_secret_key_on_conn(db):
    cursor = await db.execute("SELECT key_verify_token FROM meta")
    row = await cursor.fetchone()
    if not row:
        if settings.omnifusion_secret_key:
            try:
                f = get_fernet()
                token = f.encrypt(b"omnifusion_verification_token").decode()
                await db.execute(
                    "INSERT INTO meta (key_verify_token) VALUES (?)", (token,)
                )
                await db.commit()
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to initialize key verification token: {e}"
                )
    else:
        token = row[0]
        if token:
            if not settings.omnifusion_secret_key:
                raise ConfigurationError(
                    "OMNIFUSION_SECRET_KEY is required but not set."
                )
            try:
                f = get_fernet()
                decrypted = f.decrypt(token.encode()).decode()
                if decrypted != "omnifusion_verification_token":
                    raise ConfigurationError(
                        "Invalid OMNIFUSION_SECRET_KEY (decryption mismatch)."
                    )
            except Exception as e:
                raise ConfigurationError(
                    f"Invalid OMNIFUSION_SECRET_KEY: failed to decrypt verification token. {e}"
                )


async def check_secret_key(db=None):
    if db is not None:
        await _check_secret_key_on_conn(db)
    else:
        from ..store.db import get_db_connection

        async with get_db_connection() as conn:
            await _check_secret_key_on_conn(conn)
