import json
from pathlib import Path


def test_external_suite_registry_names_required_tier_c_harnesses():
    registry = json.loads(Path("evals/coding/external_suites.json").read_text())
    suites = {suite["id"]: suite for suite in registry["suites"]}

    assert {
        "swe-bench-lite",
        "swe-bench-verified",
        "terminal-bench",
        "internal-tool-bench",
    } <= set(suites)

    for suite in suites.values():
        assert suite["tier"] == "C"
        assert suite["status"] == "external_harness_required"
        assert suite["driver"]
        assert suite["operator_command"]
        assert suite["mock_allowed"] is False
