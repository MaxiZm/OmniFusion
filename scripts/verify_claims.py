#!/usr/bin/env python3
"""Verify the advertised-claims ledger against the codebase and website.

Gates (any failure exits non-zero):

  1. Evidence — every `covered` claim names a real implementation, docs, and
     test file, and each referenced path exists.
  2. Canonical set — the ledger defines exactly the expected claim IDs.
  3. Website cross-reference — every `data-claim-id` used on the marketing page
     exists in the ledger (when the website source is locatable).
  4. Honesty — no affirmative benchmark/Fugu *advantage* wording appears in the
     docs, README, or website unless an accepted Tier C artifact exists under
     `evals/coding/baselines/`.

Importable: `tests/test_advertised_claims.py` calls `run_checks()` directly.
Runnable: `make verify-claims` / `python scripts/verify_claims.py`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Set

CANONICAL_CLAIM_IDS: Set[str] = {
    "api-chat",
    "responses-subset",
    "streaming",
    "tool-passthrough",
    "openrouter-fusion-alias",
    "plugins",
    "providers-presets",
    "budget-ledger",
    "encrypted-keys",
    "web-grounding",
    "admin-ui",
    "traces-run-id",
    "docker-self-host",
    "security-defaults",
    "benchmark-honesty",
}

# Affirmative advantage wording. Each is forbidden in docs/README/website UNLESS
# the same line negates it (see _NEGATIONS) or an accepted Tier C artifact exists.
ADVANTAGE_PHRASES: List[str] = [
    "outperform",
    "beats ",
    "state-of-the-art",
    "sota",
    "best-in-class",
    "benchmark-leading",
    "superior to",
    "fugu replacement",
    "replaces fugu",
    "benchmark advantage",
    "faster than",
    "better than",
]

_NEGATIONS = (
    "no ",
    "not ",
    "n't",
    "without",
    "never",
    "no benchmark",
    "is not",
    "are not",
    "isn't",
    "aren't",
)

_BACKTICK = re.compile(r"`([^`]+)`")
_DATA_CLAIM = re.compile(r'data-claim-id\s*=\s*"([^"]+)"')


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ── Ledger parsing ────────────────────────────────────────────────────────────


class ClaimRow:
    def __init__(self, claim_id: str, impl: str, docs: str, tests: str, status: str):
        self.claim_id = claim_id
        self.impl = impl
        self.docs = docs
        self.tests = tests
        self.status = status

    def paths(self, cell: str) -> List[str]:
        # Backticked tokens, with any ":symbol"/":line" suffix stripped.
        return [m.split(":", 1)[0].strip() for m in _BACKTICK.findall(cell)]


def parse_ledger(ledger_path: Path) -> List[ClaimRow]:
    rows: List[ClaimRow] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 6:
            continue
        claim_match = _BACKTICK.match(cells[0])
        if not claim_match:
            continue  # header / separator rows have no backticked id
        rows.append(
            ClaimRow(
                claim_id=claim_match.group(1),
                impl=cells[2],
                docs=cells[3],
                tests=cells[4],
                status=cells[5].lower(),
            )
        )
    return rows


# ── Website source location ───────────────────────────────────────────────────


def find_web_index(root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    env_index = os.environ.get("OMNIFUSION_WEB_INDEX")
    if env_index:
        candidates.append(Path(env_index))
    env_dir = os.environ.get("OMNIFUSION_WEB_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / "index.html")
    # Default: sibling checkout next to this repo.
    candidates.append(root.parent / "OmniFusionWEB" / "index.html")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def web_claim_ids(html: str) -> Set[str]:
    return set(_DATA_CLAIM.findall(html))


# ── Honesty gate ──────────────────────────────────────────────────────────────


def has_accepted_tier_c_artifact(root: Path) -> bool:
    baselines = root / "evals" / "coding" / "baselines"
    if not baselines.is_dir():
        return False
    for path in baselines.glob("*.json"):
        if "template" in path.name.lower():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if str(data.get("tier", "")).upper() == "C":
            return True
    return False


def scan_advantage_wording(text: str) -> List[str]:
    """Return affirmative (non-negated) advantage phrases found in `text`."""
    violations: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.lower()
        for phrase in ADVANTAGE_PHRASES:
            if phrase in line and not any(neg in line for neg in _NEGATIONS):
                violations.append(f"{phrase!r} in: {raw_line.strip()[:160]}")
    return violations


# ── Checks ────────────────────────────────────────────────────────────────────


def run_checks(root: Optional[Path] = None) -> List[str]:
    root = root or repo_root()
    errors: List[str] = []

    ledger_path = root / "docs" / "advertised-claims.md"
    if not ledger_path.is_file():
        return [f"Missing ledger: {ledger_path}"]

    rows = parse_ledger(ledger_path)
    ids = [r.claim_id for r in rows]

    # Duplicate IDs.
    dupes = {cid for cid in ids if ids.count(cid) > 1}
    if dupes:
        errors.append(f"Duplicate claim IDs in ledger: {sorted(dupes)}")

    # 2. Canonical set.
    ledger_ids = set(ids)
    missing = CANONICAL_CLAIM_IDS - ledger_ids
    extra = ledger_ids - CANONICAL_CLAIM_IDS
    if missing:
        errors.append(f"Ledger missing canonical claim IDs: {sorted(missing)}")
    if extra:
        errors.append(f"Ledger has unknown claim IDs: {sorted(extra)}")

    # 1. Evidence for covered claims.
    for row in rows:
        if row.status != "covered":
            continue
        for label, cell in (("implementation", row.impl), ("docs", row.docs), ("tests", row.tests)):
            paths = row.paths(cell)
            if not paths:
                errors.append(f"[{row.claim_id}] covered but has no {label} evidence")
                continue
            for rel in paths:
                if not (root / rel).exists():
                    errors.append(
                        f"[{row.claim_id}] {label} path does not exist: {rel}"
                    )

    # 3. Website cross-reference.
    web_index = find_web_index(root)
    web_html = ""
    if web_index is not None:
        web_html = web_index.read_text(encoding="utf-8")
        used = web_claim_ids(web_html)
        unknown = used - ledger_ids
        if unknown:
            errors.append(
                f"Website uses claim IDs not in ledger: {sorted(unknown)} "
                f"(source: {web_index})"
            )

    # 4. Honesty gate.
    if not has_accepted_tier_c_artifact(root):
        scan_targets: List[Path] = [root / "README.md", ledger_path]
        docs_dir = root / "docs"
        if docs_dir.is_dir():
            scan_targets.extend(sorted(docs_dir.glob("*.md")))
        seen: Set[Path] = set()
        for target in scan_targets:
            if target in seen or not target.is_file():
                continue
            seen.add(target)
            for v in scan_advantage_wording(target.read_text(encoding="utf-8")):
                errors.append(f"Affirmative advantage wording in {target.name}: {v}")
        if web_html:
            for v in scan_advantage_wording(web_html):
                errors.append(f"Affirmative advantage wording in website: {v}")

    return errors


def main(argv: Optional[List[str]] = None) -> int:
    errors = run_checks()
    if errors:
        print("Claims verification FAILED:\n")
        for err in errors:
            print(f"  ✗ {err}")
        print(f"\n{len(errors)} problem(s).")
        return 1
    print("Claims verification passed: ledger, evidence, website, and honesty gates OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
