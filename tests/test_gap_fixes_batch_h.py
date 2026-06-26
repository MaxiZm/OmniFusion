"""Batch H: eval preflight/provenance + admin full-PresetV2 authoring."""
import json
from pathlib import Path

from omnifusion.evals import coding
from omnifusion.fusion.types import Preset, PresetPrompts, PresetStage


def test_expected_passthrough_model_strips_openai_prefix():
    assert (
        coding.expected_passthrough_model({"model": "openai/fusion/general"})
        == "fusion/general"
    )
    assert coding.expected_passthrough_model({"model": "fusion/general"}) == "fusion/general"


def test_task_suite_checksum_is_content_addressed(tmp_path):
    a = tmp_path / "a.json"
    a.write_text('[{"id": "x"}]')
    checksum = coding.task_suite_checksum(a)
    assert checksum.startswith("sha256:")
    # Stable for identical content, different for changed content.
    assert checksum == coding.task_suite_checksum(a)
    a.write_text('[{"id": "y"}]')
    assert checksum != coding.task_suite_checksum(a)


def test_smoke_payload_records_passthrough_provenance():
    config = coding.load_json(coding.DEFAULT_CONFIG)
    payload = coding.build_payload("coding-smoke", config, [], mock=True)
    assert payload["provenance"]["expected_passthrough_model"] == "fusion/general"


def test_baseline_template_has_full_provenance_and_best_single():
    template = json.loads(Path("evals/coding/baselines/baseline_template.json").read_text())
    prov = template["required_provenance"]
    for field in ("provider", "date", "pricing", "task_suite_checksum"):
        assert field in prov, f"baseline provenance missing {field}"
    assert "best_single_configured_model" in template["comparison"]

    table = Path("evals/coding/baselines/baseline_table.md").read_text()
    assert "best single configured model" in table


def test_admin_preset_form_exposes_full_v2_fields():
    form = Path("src/omnifusion/web/templates/presets.html").read_text()
    for field in (
        'name="display_name"',
        'name="mode"',
        'name="web_enabled"',
        'name="prompt_global"',
        'name="prompt_panel"',
        'name="prompt_judge"',
        'name="prompt_final"',
    ):
        assert field in form, f"admin preset form missing {field}"
    # Conductor is selectable (experimental) now that the route accepts it.
    assert 'value="conductor"' in form


def test_full_v2_preset_authored_via_admin_fields_roundtrips():
    """Mirrors what save_preset_route builds from the form, proving the console can
    author a complete PresetV2 (display_name/mode/web_enabled/prompts)."""
    preset = Preset(
        name="authored",
        display_name="Authored Council",
        mode="fusion",
        strategy="B",
        web_enabled=True,
        prompts=PresetPrompts(
            global_prompt="be concise",
            role_prompts={"panel": "draft", "judge": "score", "final": "merge"},
        ),
        panel_models=["m"],
        panel=PresetStage(max_tokens=16, timeout=5),
        judge_model="m",
        judge=PresetStage(max_tokens=16, timeout=5),
        final_model="m",
        final=PresetStage(max_tokens=16, timeout=5),
        cost_ceiling=0.5,
    )
    reloaded = Preset.model_validate_json(preset.model_dump_json())
    assert reloaded.display_name == "Authored Council"
    assert reloaded.web_enabled is True
    assert reloaded.prompts.global_prompt == "be concise"
    assert reloaded.prompts.role_prompts["judge"] == "score"


def test_mock_full_run_is_not_labeled_tier_c():
    """[P2] A mocked coding-full payload must not claim Tier C evidence."""
    config = coding.load_json(coding.DEFAULT_CONFIG)
    mock_payload = coding.build_payload("coding-full", config, [], mock=True)
    assert mock_payload["tier"] == "mock"
    real_payload = coding.build_payload("coding-full", config, [], mock=False)
    assert real_payload["tier"] == "C"


def _run_suite_args(tmp_path, *, mock, fail_under):
    import argparse

    tasks = tmp_path / "tasks.json"
    tasks.write_text('[{"id": "t1", "language": "python", "prompt": "x", "mock_passed": false}]')
    return argparse.Namespace(
        suite="smoke",
        config=coding.DEFAULT_CONFIG,
        tasks=tasks,
        output=tmp_path / "out.json",
        mock=mock,
        timeout_s=5,
        fail_under=fail_under,
        func=None,
    )


def test_fail_under_gate_is_exempt_for_mock_runs(tmp_path):
    """A mock run with a high threshold still exits 0 (it's a contract check)."""
    assert coding.run_suite(_run_suite_args(tmp_path, mock=True, fail_under=0.9)) == 0


def test_fail_under_gate_returns_one_on_real_run_below_threshold(tmp_path, monkeypatch):
    """[P3] A REAL (non-mock) run_suite returns 1 when the pass rate is below the
    threshold, and 0 when it meets it. Patches the live Aider/preflight dependencies
    so the gate logic itself is exercised offline."""
    monkeypatch.setattr(coding, "preflight_model_passthrough", lambda config: None)

    def failing_task(config, task, timeout_s):
        return {
            "id": task["id"],
            "language": task["language"],
            "passed": False,
            "cost_usd": 0.0,
            "wall_time_s": 0.0,
            "driver": "aider",
            "validation": {"passed": False, "checks": []},
        }

    monkeypatch.setattr(coding, "run_aider_task", failing_task)
    # 0/1 passed, threshold 0.5 -> gate trips -> exit 1.
    assert coding.run_suite(_run_suite_args(tmp_path, mock=False, fail_under=0.5)) == 1

    def passing_task(config, task, timeout_s):
        result = failing_task(config, task, timeout_s)
        result["passed"] = True
        return result

    monkeypatch.setattr(coding, "run_aider_task", passing_task)
    # 1/1 passed, threshold 0.5 -> gate passes -> exit 0.
    assert coding.run_suite(_run_suite_args(tmp_path, mock=False, fail_under=0.5)) == 0
    # No threshold -> exit 0 even if it had failed (report-generation default).
    monkeypatch.setattr(coding, "run_aider_task", failing_task)
    assert coding.run_suite(_run_suite_args(tmp_path, mock=False, fail_under=None)) == 0
