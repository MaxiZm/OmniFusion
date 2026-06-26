import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.presets import save_preset


def test_aider_eval_config_is_pinned_and_openai_compatible():
    config = json.loads(Path("evals/coding/aider_config.json").read_text())

    assert config["aider_chat_version"] == "0.86.2"
    assert config["base_url_env"] == "OPENAI_API_BASE"
    assert config["api_key_env"] == "OPENAI_API_KEY"
    assert config["base_url"] == "http://localhost:8000/v1"
    assert config["model"] == "openai/fusion/general"
    assert config["docs"]["openai_compat"] == "https://aider.chat/docs/llms/openai-compat.html"
    assert config["docs"]["leaderboard"] == "https://aider.chat/docs/leaderboards/"


@pytest.mark.asyncio
async def test_openai_prefixed_fusion_model_resolves_to_preset(tmp_path, monkeypatch):
    import omnifusion.api.chat as chat_mod
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m1b.db")
    settings.omnifusion_api_keys = ["eval-key"]

    captured = {}

    async def fake_run_fusion(run_id, preset, body, key_hash):
        captured["preset_name"] = preset.name
        captured["body_model"] = body.model
        return {
            "id": "chatcmpl-m1b",
            "object": "chat.completion",
            "created": 1,
            "model": f"fusion/{preset.name}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    try:
        await init_db()
        stage = PresetStage(max_tokens=8, timeout=5)
        await save_preset(
            Preset(
                name="general",
                strategy="B",
                panel_models=["m"],
                panel=stage,
                judge_model="m",
                judge=stage,
                final_model="m",
                final=stage,
                min_panel_success=1,
            )
        )
        monkeypatch.setattr(chat_mod, "run_fusion", fake_run_fusion)

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer eval-key"},
                json={
                    "model": "openai/fusion/general",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status_code == 200
        assert response.json()["model"] == "fusion/general"
        assert captured == {"preset_name": "general", "body_model": "fusion/general"}
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys


def test_eval_coding_smoke_mock_outputs_raw_metrics(tmp_path):
    output_path = tmp_path / "smoke.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnifusion.evals.coding",
            "smoke",
            "--mock",
            "--output",
            str(output_path),
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": "src"},
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text())
    assert payload["suite"] == "coding-smoke"
    assert payload["aider_chat_version"] == "0.86.2"
    assert len(payload["tasks"]) <= 20
    required_fields = {"id", "passed", "cost_usd", "wall_time_s"}
    assert all(required_fields <= set(task) for task in payload["tasks"])
    assert "pass_rate" in payload["raw"]
    assert "total_cost_usd" in payload["raw"]
    assert "total_wall_time_s" in payload["raw"]
    assert "95% CI" not in json.dumps(payload)
    assert "coding-smoke" in result.stdout
