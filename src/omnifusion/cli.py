import sys
import os
import asyncio
import json
import secrets
import yaml
from cryptography.fernet import Fernet
from .store.db import get_db_connection


def genkey():
    key = Fernet.generate_key().decode()
    print("Your new OMNIFUSION_SECRET_KEY:")
    print(key)
    print(
        "\nWARNING: Keep this key safe. If you lose it, you will not be able to decrypt your stored provider API keys."
    )


# --- QuickStart ------------------------------------------------------------
# One command that takes a fresh checkout to a server that boots: provision the
# three required secrets in .env (without clobbering real values an operator has
# already set), then create the SQLite database and key-verification token.

ENV_PATH = ".env"
ENV_EXAMPLE_PATH = ".env.example"

# Values that mean "still needs to be filled in". Compared case-insensitively
# after stripping. The secret-key/admin-password sets mirror settings.py; the
# .env.example client-key sample is `key_one,key_two`.
_PLACEHOLDER_SECRET_KEYS = {"", "your_fernet_secret_key", "your-fernet-secret-key", "changeme"}
_PLACEHOLDER_ADMIN_PASSWORDS = {"", "your_admin_password", "your-admin-password", "changeme"}
_PLACEHOLDER_API_KEYS = {"", "key_one", "key_two", "key1", "key2", "your_api_key"}


def _is_placeholder_secret_key(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDER_SECRET_KEYS


def _is_placeholder_admin_password(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDER_ADMIN_PASSWORDS


def _is_placeholder_api_keys(value: str) -> bool:
    keys = [k.strip().lower() for k in value.split(",") if k.strip()]
    return not keys or all(k in _PLACEHOLDER_API_KEYS for k in keys)


def _generate_client_key() -> str:
    return "sk-omnifusion-" + secrets.token_urlsafe(24)


def _generate_admin_password() -> str:
    return secrets.token_urlsafe(18)


def _read_env_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return f.read().splitlines()


def _get_env_value(lines: list[str], key: str) -> str | None:
    """Return the raw value for KEY=VALUE in the .env lines, or None if absent."""
    prefix = f"{key}="
    for line in lines:
        if line.lstrip().startswith(prefix) and not line.lstrip().startswith("#"):
            return line.split("=", 1)[1]
    return None


def _set_env_value(lines: list[str], key: str, value: str) -> list[str]:
    """Set KEY=VALUE in-place if present, otherwise append it. Preserves comments
    and ordering of every other line so a hand-edited .env survives untouched."""
    prefix = f"{key}="
    out = []
    replaced = False
    for line in lines:
        if line.lstrip().startswith(prefix) and not line.lstrip().startswith("#"):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    return out


def _write_env_lines(path: str, lines: list[str]) -> None:
    # .env holds secrets — write it owner-read/write only (0o600), matching the
    # treatment of `omnifusion export`.
    old_umask = os.umask(0o177)
    try:
        with open(path, "w") as f:
            f.write("\n".join(lines).rstrip("\n") + "\n")
    finally:
        os.umask(old_umask)


def _reload_settings() -> None:
    """Re-read .env into the in-process settings singleton so the just-written
    secret key is live for init_db (which writes the key-verification token)."""
    from .settings import settings, Settings

    settings.__dict__.update(Settings().__dict__)


def quickstart(serve: bool = False) -> None:
    from .settings import validate_startup_security
    from .secrets.crypto import generate_key
    from .store.db import init_db

    print("=" * 60)
    print("OmniFusion QuickStart")
    print("=" * 60)

    # 1. Ensure a .env exists, seeded from .env.example when available.
    created_env = False
    if not os.path.exists(ENV_PATH):
        if os.path.exists(ENV_EXAMPLE_PATH):
            lines = _read_env_lines(ENV_EXAMPLE_PATH)
            print(f"Created {ENV_PATH} from {ENV_EXAMPLE_PATH}")
        else:
            lines = []
            print(f"Created {ENV_PATH}")
        created_env = True
    else:
        lines = _read_env_lines(ENV_PATH)
        print(f"Using existing {ENV_PATH}")

    # 2. Fill the three required secrets only when missing/placeholder, so an
    #    operator's real values are never overwritten.
    generated_client_key = None

    secret_key = _get_env_value(lines, "OMNIFUSION_SECRET_KEY") or ""
    if _is_placeholder_secret_key(secret_key):
        lines = _set_env_value(lines, "OMNIFUSION_SECRET_KEY", generate_key())
        print("  OMNIFUSION_SECRET_KEY     generated")
    else:
        print("  OMNIFUSION_SECRET_KEY     kept (already set)")

    admin_password = _get_env_value(lines, "OMNIFUSION_ADMIN_PASSWORD") or ""
    if _is_placeholder_admin_password(admin_password):
        new_password = _generate_admin_password()
        lines = _set_env_value(lines, "OMNIFUSION_ADMIN_PASSWORD", new_password)
        print(f"  OMNIFUSION_ADMIN_PASSWORD generated -> {new_password}")
    else:
        print("  OMNIFUSION_ADMIN_PASSWORD kept (already set)")

    api_keys = _get_env_value(lines, "OMNIFUSION_API_KEYS") or ""
    if _is_placeholder_api_keys(api_keys):
        generated_client_key = _generate_client_key()
        lines = _set_env_value(lines, "OMNIFUSION_API_KEYS", generated_client_key)
        print(f"  OMNIFUSION_API_KEYS       generated -> {generated_client_key}")
    else:
        print("  OMNIFUSION_API_KEYS       kept (already set)")

    _write_env_lines(ENV_PATH, lines)

    # 3. Reload settings from the new .env, validate, and initialize the DB.
    _reload_settings()
    try:
        validate_startup_security()
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    asyncio.run(init_db())
    from .settings import settings as _settings

    print(f"Initialized database at {_settings.db_path}")

    # 4. Summary + copy-pasteable next steps.
    print("\n" + "=" * 60)
    print("Ready. Next steps:")
    print("=" * 60)
    if generated_client_key:
        print(f"\nClient API key:  {generated_client_key}")
        print("  (also stored in OMNIFUSION_API_KEYS in .env)")
    print("\n1. Start the server:")
    print("     make dev            # or: omnifusion quickstart --serve")
    print("\n2. Open the admin UI to register a provider + preset:")
    print("     http://127.0.0.1:8000/admin")
    print("\n3. Call chat completions:")
    sample_key = generated_client_key or "$OMNIFUSION_API_KEY"
    print("     curl http://127.0.0.1:8000/v1/chat/completions \\")
    print(f'       -H "Authorization: Bearer {sample_key}" \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"model": "fusion/general", "messages": '
          '[{"role": "user", "content": "Hello"}]}\'')
    if created_env or generated_client_key:
        print("\nKeep .env private — it now contains live secrets (perms 0o600).")

    # 5. Optionally boot the dev server right away.
    if serve:
        print("\nStarting development server on http://127.0.0.1:8000 ...")
        import uvicorn

        uvicorn.run("src.omnifusion.main:app", host="127.0.0.1", port=8000, reload=True)


async def rotate_key_async():
    old_key_str = os.getenv("OMNIFUSION_SECRET_KEY_OLD")
    new_key_str = os.getenv("OMNIFUSION_SECRET_KEY")

    if not old_key_str or not new_key_str:
        print(
            "ERROR: Both OMNIFUSION_SECRET_KEY_OLD and OMNIFUSION_SECRET_KEY environment variables must be set."
        )
        sys.exit(1)

    try:
        old_f = Fernet(old_key_str.encode())
    except Exception as e:
        print(f"ERROR: Invalid OMNIFUSION_SECRET_KEY_OLD: {e}")
        sys.exit(1)

    try:
        new_f = Fernet(new_key_str.encode())
    except Exception as e:
        print(f"ERROR: Invalid OMNIFUSION_SECRET_KEY: {e}")
        sys.exit(1)

    print("Starting key rotation...")
    async with get_db_connection() as db:
        try:
            await db.execute("BEGIN IMMEDIATE")

            # Read all provider rows
            cursor = await db.execute("SELECT id, enc_key FROM providers")
            rows = await cursor.fetchall()

            for pid, enc_key in rows:
                if enc_key:
                    try:
                        # Decrypt
                        plain = old_f.decrypt(enc_key).decode()
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to decrypt key for provider {pid} with old key. Aborting. Error: {e}"
                        )
                    # Encrypt with new key
                    new_enc = new_f.encrypt(plain.encode())
                    await db.execute(
                        "UPDATE providers SET enc_key = ? WHERE id = ?", (new_enc, pid)
                    )

            # Update verification token in meta
            new_verify_token = new_f.encrypt(b"omnifusion_verification_token").decode()
            # Delete old verify tokens and insert new one
            await db.execute("DELETE FROM meta")
            await db.execute(
                "INSERT INTO meta (key_verify_token) VALUES (?)", (new_verify_token,)
            )

            await db.commit()
            print(
                "SUCCESS: Key rotation completed successfully! All stored keys re-encrypted."
            )
        except Exception as e:
            await db.execute("ROLLBACK")
            print(f"ERROR: Key rotation failed. Rolled back changes. Detail: {e}")
            sys.exit(1)


async def purge_expired_runs():
    import time

    async with get_db_connection() as db:
        now = int(time.time())
        cursor = await db.execute("DELETE FROM runs WHERE expires_at < ?", (now,))
        await db.commit()
        print(f"Purged {cursor.rowcount} expired runs.")


async def export_db(filepath: str = "omnifusion.yaml"):
    """
    Fix #9: Export writes decrypted provider API keys. We now:
    1. Print a prominent security warning before writing.
    2. Set file permissions to owner-read-only (0o600) using umask.
    """
    # Print security warning BEFORE writing the file
    print("=" * 60)
    print("SECURITY WARNING")
    print("=" * 60)
    print(f"The export file '{filepath}' will contain DECRYPTED provider")
    print("API keys in plaintext. Treat this file as highly sensitive.")
    print("Store it securely and delete it when no longer needed.")
    print("=" * 60)

    async with get_db_connection() as db:
        # 1. Export providers
        cursor = await db.execute(
            "SELECT id, type, base_url, enc_key, api_key_ref, models_json FROM providers"
        )
        providers = []
        for row in await cursor.fetchall():
            pid, ptype, base_url, enc_key, api_key_ref, models_json = row
            plain_key = ""
            if enc_key:
                from .secrets.crypto import decrypt_key

                try:
                    plain_key = decrypt_key(enc_key)
                except Exception:
                    plain_key = "[Could not decrypt]"
            providers.append(
                {
                    "id": pid,
                    "type": ptype,
                    "base_url": base_url,
                    "api_key": plain_key,
                    "api_key_ref": api_key_ref,
                    "models": json.loads(models_json) if models_json else [],
                }
            )

        # 2. Export presets
        cursor = await db.execute("SELECT spec_json FROM presets")
        presets = []
        for row in await cursor.fetchall():
            presets.append(json.loads(row[0]))

        data = {"providers": providers, "presets": presets}

        # Fix #9: Write file with restrictive permissions (owner-read-only = 0o600)
        # by temporarily tightening the umask.
        old_umask = os.umask(0o177)  # 0o177 = block group+other read/write/exec
        try:
            with open(filepath, "w") as f:
                yaml.safe_dump(data, f)
        finally:
            os.umask(old_umask)

        print(f"Successfully exported configuration to '{filepath}' (permissions: 0o600)")
        print("DELETE this file after importing it to a new instance.")


async def import_db(filepath: str = "omnifusion.yaml"):
    if not os.path.exists(filepath):
        print(f"ERROR: File {filepath} not found.")
        sys.exit(1)

    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    providers = data.get("providers", [])
    presets = data.get("presets", [])

    # Save providers
    from .store.providers import save_provider

    for p in providers:
        api_key = p.get("api_key", "")
        if api_key == "[Could not decrypt]":
            api_key = ""
        await save_provider(
            provider_id=p["id"],
            p_type=p["type"],
            plain_key=api_key,
            base_url=p.get("base_url"),
            api_key_ref=p.get("api_key_ref"),
            models=p.get("models", []),
        )

    # Save presets
    from .store.presets import save_preset
    from .fusion.types import Preset

    for pr in presets:
        preset_obj = Preset.model_validate(pr)
        await save_preset(preset_obj)

    print(
        f"Successfully imported {len(providers)} providers and {len(presets)} presets from {filepath}"
    )


def _read_json_or_yaml(filepath: str):
    with open(filepath, "r") as f:
        if filepath.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


async def preset_save_async(filepath: str):
    from .fusion.types import Preset
    from .store.presets import save_preset

    preset = Preset.model_validate(_read_json_or_yaml(filepath))
    await save_preset(preset)
    print(f"Saved preset {preset.name}")


async def preset_get_async(name: str):
    from .store.presets import get_preset

    preset = await get_preset(name)
    if not preset:
        print(f"ERROR: Preset {name} not found.")
        sys.exit(1)
    print(preset.model_dump_json(indent=2))


async def preset_list_async():
    from .store.presets import list_presets

    presets = await list_presets()
    print(json.dumps([preset.model_dump() for preset in presets], indent=2))


async def preset_delete_async(name: str):
    from .store.presets import delete_preset

    await delete_preset(name)
    print(f"Deleted preset {name}")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: omnifusion [quickstart [--serve] | genkey | rotate-key | purge | export [file] | import [file] | preset ...]"
        )
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "quickstart":
        quickstart(serve="--serve" in sys.argv[2:])
    elif cmd == "genkey":
        genkey()
    elif cmd == "rotate-key":
        asyncio.run(rotate_key_async())
    elif cmd == "purge":
        asyncio.run(purge_expired_runs())
    elif cmd == "export":
        file = sys.argv[2] if len(sys.argv) > 2 else "omnifusion.yaml"
        asyncio.run(export_db(file))
    elif cmd == "import":
        file = sys.argv[2] if len(sys.argv) > 2 else "omnifusion.yaml"
        asyncio.run(import_db(file))
    elif cmd == "preset":
        if len(sys.argv) < 3:
            print("Usage: omnifusion preset [list | get NAME | save FILE | delete NAME]")
            sys.exit(1)
        subcmd = sys.argv[2]
        if subcmd == "list":
            asyncio.run(preset_list_async())
        elif subcmd == "get" and len(sys.argv) >= 4:
            asyncio.run(preset_get_async(sys.argv[3]))
        elif subcmd == "save" and len(sys.argv) >= 4:
            asyncio.run(preset_save_async(sys.argv[3]))
        elif subcmd == "delete" and len(sys.argv) >= 4:
            asyncio.run(preset_delete_async(sys.argv[3]))
        else:
            print("Usage: omnifusion preset [list | get NAME | save FILE | delete NAME]")
            sys.exit(1)
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: omnifusion [quickstart | genkey | rotate-key | purge | export | import | preset]")
        sys.exit(1)


if __name__ == "__main__":
    main()
