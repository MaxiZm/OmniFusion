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
    for client, cells in matrix["matrix"].items():
        assert set(cells) == REQUIRED_CELLS
        for cell, result in cells.items():
            assert result["status"] in {"covered", "unsupported", "manual"}
            if result["status"] == "covered":
                assert result["covered_by"], f"{client}/{cell} missing test reference"
            else:
                assert result["reason"], f"{client}/{cell} missing reason"


def test_api_compatibility_doc_publishes_responses_subset_and_alias_status():
    doc = Path("docs/api-compatibility.md").read_text()

    assert "minimal text-compatible" in doc
    assert "/v1/responses" in doc
    assert "response.output_text.delta" in doc
    assert "response.completed" in doc
    assert "previous_response_id" in doc
    assert "compat_placeholder - not conductor-backed yet" in doc
