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
