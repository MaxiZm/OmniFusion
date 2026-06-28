#!/usr/bin/env python3
"""OpenAI Python SDK compatibility smoke against a running OmniFusion instance.

Opt-in / live only. Skips cleanly (exit 0) when the endpoint env vars are unset,
so it is safe to wire into a Makefile and CI. It asserts wire compatibility — the
canonical OpenAI response shape, a run-id header, and streaming — never any
benchmark or quality claim.

    export OMNIFUSION_BASE_URL=http://127.0.0.1:8000/v1
    export OMNIFUSION_API_KEY=your-omnifusion-client-key
    uv run python scripts/compat/openai_python.py
"""

import os
import sys


def main() -> int:
    base_url = os.environ.get("OMNIFUSION_BASE_URL")
    api_key = os.environ.get("OMNIFUSION_API_KEY")
    model = os.environ.get("OMNIFUSION_MODEL", "fusion/general")

    if not base_url or not api_key:
        print(
            "[skip] OMNIFUSION_BASE_URL / OMNIFUSION_API_KEY not set; "
            "this is an opt-in live smoke. Nothing to do."
        )
        return 0

    try:
        from openai import OpenAI
    except ImportError:
        print("[skip] openai package not installed (pip install openai).")
        return 0

    client = OpenAI(base_url=base_url, api_key=api_key)

    # 1. Non-streaming completion + raw headers (run-id correlation).
    raw = client.chat.completions.with_raw_response.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: pong."}],
        max_tokens=64,
    )
    headers = {k.lower(): v for k, v in raw.headers.items()}
    assert "x-omnifusion-run-id" in headers, "missing X-OmniFusion-Run-Id header"
    completion = raw.parse()
    assert completion.choices, "no choices returned"
    content = completion.choices[0].message.content
    assert content is not None, "no content in completion"
    print(f"[ok] non-stream  model={completion.model} run_id={headers['x-omnifusion-run-id']}")
    print(f"     content: {content[:80]!r}")
    if completion.usage is not None:
        print(
            f"     usage: prompt={completion.usage.prompt_tokens} "
            f"completion={completion.usage.completion_tokens}"
        )

    # 2. Streaming completion with usage.
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count: one two three."}],
        max_tokens=64,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = 0
    streamed = ""
    for event in stream:
        chunks += 1
        if event.choices and event.choices[0].delta.content:
            streamed += event.choices[0].delta.content
    assert chunks > 0, "stream produced no chunks"
    print(f"[ok] streaming   chunks={chunks} content={streamed[:80]!r}")

    print("\nOpenAI Python SDK compatibility smoke passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)
