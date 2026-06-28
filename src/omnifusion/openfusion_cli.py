from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _api_key(args) -> str:
    key = args.api_key or os.getenv("OMNIFUSION_API_KEY")
    if not key:
        print("ERROR: Set OMNIFUSION_API_KEY or pass --api-key.", file=sys.stderr)
        raise SystemExit(1)
    return key


def _post_chat(args, prompt: str) -> dict:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if args.web:
        body["tools"] = [{"type": "openrouter:web_search"}]
    if args.preset:
        body["openfusion"] = {"preset": args.preset}
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={
            "authorization": f"Bearer {_api_key(args)}",
            "content-type": "application/json",
            "accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        print(f"ERROR: HTTP {exc.code}: {detail}", file=sys.stderr)
        raise SystemExit(1) from exc


def cmd_setup(args) -> None:
    from omnifusion.cli import quickstart

    quickstart(serve=args.serve)


def cmd_web(args) -> None:
    import uvicorn

    uvicorn.run("omnifusion.main:app", host=args.host, port=args.port, reload=args.reload)


def cmd_ask(args) -> None:
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        print("ERROR: ask requires a prompt.", file=sys.stderr)
        raise SystemExit(1)
    payload = _post_chat(args, prompt)
    print(payload["choices"][0]["message"]["content"] or "")


def cmd_chat(args) -> None:
    print("OpenFusion chat. Press Ctrl-D to exit.")
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            return
        if not prompt:
            continue
        payload = _post_chat(args, prompt)
        print(payload["choices"][0]["message"]["content"] or "")


def _add_common_api_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default=os.getenv("OMNIFUSION_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="openfusion")
    parser.add_argument("--preset", choices=["quality", "budget"], default=None)
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openfusion")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup")
    setup.add_argument("--serve", action="store_true")
    setup.set_defaults(func=cmd_setup)

    web = sub.add_parser("web")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--reload", action="store_true")
    web.set_defaults(func=cmd_web)

    ask = sub.add_parser("ask")
    _add_common_api_args(ask)
    ask.add_argument("prompt", nargs="+")
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat")
    _add_common_api_args(chat)
    chat.set_defaults(func=cmd_chat)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
