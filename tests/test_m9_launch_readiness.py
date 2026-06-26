from pathlib import Path

import yaml


def test_docker_compose_binds_localhost_by_default():
    compose = yaml.safe_load(Path("deploy/docker-compose.yml").read_text())
    ports = compose["services"]["omnifusion"]["ports"]

    assert "127.0.0.1:8000:8000" in ports


def test_launch_readiness_docs_and_ci_are_present():
    required_files = [
        ".github/workflows/ci.yml",
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
