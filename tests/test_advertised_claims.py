"""Claim-ledger gate: evidence exists, canonical IDs are complete, website claim IDs
cross-reference, and no affirmative benchmark/Fugu advantage wording slips in."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import verify_claims  # noqa: E402


def test_ledger_and_gates_pass():
    errors = verify_claims.run_checks(ROOT)
    assert errors == [], "Claims verification failed:\n" + "\n".join(errors)


def test_ledger_defines_exactly_the_canonical_claims():
    rows = verify_claims.parse_ledger(ROOT / "docs" / "advertised-claims.md")
    ids = {r.claim_id for r in rows}
    assert ids == verify_claims.CANONICAL_CLAIM_IDS


def test_every_covered_claim_cites_evidence():
    rows = verify_claims.parse_ledger(ROOT / "docs" / "advertised-claims.md")
    covered = [r for r in rows if r.status == "covered"]
    assert covered, "expected at least one covered claim"
    for row in covered:
        assert row.paths(row.impl), f"{row.claim_id} missing implementation evidence"
        assert row.paths(row.tests), f"{row.claim_id} missing test evidence"


def test_advantage_wording_scan_is_negation_aware():
    # Affirmative claim → flagged.
    assert verify_claims.scan_advantage_wording("OmniFusion outperforms GPT-4 on SWE-bench")
    # Negated claim (our honesty stance) → allowed.
    assert not verify_claims.scan_advantage_wording(
        "It is not a benchmark-backed Fugu replacement yet."
    )
    assert not verify_claims.scan_advantage_wording(
        "No benchmark advantage is claimed from mocked tests."
    )


def test_website_claim_ids_are_a_subset_of_the_ledger():
    web_index = verify_claims.find_web_index(ROOT)
    if web_index is None:
        import pytest

        pytest.skip("OmniFusionWEB checkout not found; cross-reference skipped")
    used = verify_claims.web_claim_ids(web_index.read_text(encoding="utf-8"))
    rows = verify_claims.parse_ledger(ROOT / "docs" / "advertised-claims.md")
    ledger_ids = {r.claim_id for r in rows}
    assert used, "website declares no data-claim-id attributes"
    unknown = used - ledger_ids
    assert not unknown, f"website uses claim IDs absent from ledger: {sorted(unknown)}"
