from pathlib import Path
import tomllib

import yaml


def test_docker_compose_binds_localhost_by_default():
    compose = yaml.safe_load(Path("deploy/docker-compose.yml").read_text())
    ports = compose["services"]["omnifusion"]["ports"]

    assert "127.0.0.1:8000:8000" in ports


def test_launch_readiness_docs_and_ci_are_present():
    required_files = [
        ".github/workflows/ci.yml",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "docs/benchmark-reproduction.md",
        "docs/budgeting-tracing.md",
        "docs/fugu-architecture.md",
        "docs/providers-presets.md",
        "docs/security-model.md",
        "docs/quickstart.md",
    ]
    for file_name in required_files:
        assert Path(file_name).exists(), f"missing {file_name}"

    ci = Path(".github/workflows/ci.yml").read_text()
    makefile = Path("Makefile").read_text()
    assert "make lint" in ci
    assert "make test" in ci
    assert "make install-smoke" in ci
    assert "make security-audit" in ci
    assert "docker build" in ci
    assert "install-smoke:" in makefile
    assert "security-audit:" in makefile

    security = Path("docs/security-model.md").read_text()
    assert "web_fetch" in security
    assert "OMNIFUSION_ALLOW_PRIVATE_EGRESS" in security
    assert "No benchmark advantage claim" in security

    benchmark = Path("docs/benchmark-reproduction.md").read_text()
    assert "Tier C" in benchmark
    assert "best single configured model" in benchmark
    assert "judge-selected best-of-N" in benchmark

    fugu = Path("docs/fugu-architecture.md").read_text()
    assert "transparent approximation" in fugu
    assert "ablation_required" in fugu
    assert "off by default" in fugu


def test_readme_has_ci_badge_and_from_readme_walkthrough():
    readme = Path("README.md").read_text()
    # Real CI badge wired to the actual workflow.
    assert "actions/workflows/ci.yml/badge.svg" in readme
    # The exit-gate 'works FROM README' flow: configure -> create/confirm -> call.
    assert "omnifusion genkey" in readme
    assert "preset list" in readme
    assert "fugu-ultra" in readme
    assert "/v1/chat/completions" in readme
    # Links operators to the docs set.
    assert "docs/quickstart.md" in readme


def test_changelog_documents_initial_release():
    changelog = Path("CHANGELOG.md").read_text()
    assert "0.1.0" in changelog
    assert "Keep a Changelog" in changelog


def test_package_metadata_is_publishable():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert project["description"] != "Add your description here"
    assert "OpenAI-compatible" in project["description"]
    assert project["license"] == "Apache-2.0"
    assert project["urls"]["Repository"] == "https://github.com/MaxiZm/OmniFusion"


def test_local_opencode_config_is_not_committed_with_inline_keys():
    gitignore = Path(".gitignore").read_text()
    example = Path("opencode.example.json").read_text()

    assert "opencode.json" in gitignore
    assert "opencode.example.json" not in gitignore
    assert "OMNIFUSION_API_KEY" in example
    assert '"apiKey"' not in example
    assert "sk-" not in example
