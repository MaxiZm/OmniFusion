import sys
import os
import asyncio
import json
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


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: omnifusion [genkey | rotate-key | purge | export [file] | import [file]]"
        )
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "genkey":
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
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: omnifusion [genkey | rotate-key | purge | export | import]")
        sys.exit(1)


if __name__ == "__main__":
    main()
