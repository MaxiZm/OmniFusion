import json
from pathlib import Path


REQUIRED_CLIENTS = {
    "openai-python",
    "openai-node",
    "aider",
    "opencode",
    "continue",
    "cursor-openai-compat",
}
REQUIRED_CELLS = {
    "chat",
    "stream",
    "stream_usage",
    "tools",
    "tool_stream",
    "errors",
    "models",
}


def test_client_contract_matrix_has_no_unclassified_cells():
    matrix = json.loads(Path("docs/client-contract-matrix.json").read_text())

    assert set(matrix["clients"]) == REQUIRED_CLIENTS
    for client, pin in matrix["pins"].items():
        assert "pending" not in pin.lower(), f"{client} pin is pending"
    for client, cells in matrix["matrix"].items():
        assert set(cells) == REQUIRED_CELLS
        for cell, result in cells.items():
            assert result["status"] in {"covered", "unsupported", "manual"}
            if result["status"] == "covered":
                assert result["covered_by"], f"{client}/{cell} missing test reference"
                # The cited test(s) must actually exist on disk — a green status may
                # not point at a non-existent file.
                for test_ref in result["covered_by"]:
                    ref_path = Path(test_ref.split("::", 1)[0])
                    assert ref_path.exists(), f"{client}/{cell} cites missing {test_ref}"
                # Honesty: every covered cell declares whether it is real-SDK or
                # wire-contract coverage, and an "sdk" claim must cite an SDK test.
                assert result["coverage_type"] in {"sdk", "wire_contract"}, (
                    f"{client}/{cell} has invalid coverage_type"
                )
                if result["coverage_type"] == "sdk":
                    assert any(
                        "client_" in ref or "_client" in ref for ref in result["covered_by"]
                    ), f"{client}/{cell} claims sdk coverage without an SDK test"
            else:
                assert result["reason"], f"{client}/{cell} missing reason"


def test_stream_usage_cells_cite_an_include_usage_test():
    """The stream_usage cells must be backed by a test that actually exercises
    stream_options.include_usage, not a generic streaming test."""
    matrix = json.loads(Path("docs/client-contract-matrix.json").read_text())
    backing_test = Path("tests/test_stream_usage.py")
    assert backing_test.exists()
    assert "include_usage" in backing_test.read_text()

    for client, cells in matrix["matrix"].items():
        cell = cells["stream_usage"]
        if cell["status"] == "covered":
            assert "tests/test_stream_usage.py" in cell["covered_by"], (
                f"{client}/stream_usage must cite the include_usage test"
            )


def test_api_compatibility_doc_publishes_responses_subset_and_alias_status():
    doc = Path("docs/api-compatibility.md").read_text()

    assert "minimal text-compatible" in doc
    assert "/v1/responses" in doc
    assert "response.output_text.delta" in doc
    assert "response.completed" in doc
    assert "previous_response_id" in doc
    assert "compat_placeholder - not conductor-backed yet" in doc
    assert "ablation-proven" in doc
