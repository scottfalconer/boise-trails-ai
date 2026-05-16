#!/usr/bin/env python3
"""Run an optional local Codex hiker-style route review for a route pack."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_PROMPT = YEAR_DIR / "prompts" / "human_route_reviewer.md"
DEFAULT_SCHEMA = YEAR_DIR / "schemas" / "route_review.schema.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-reviews"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def route_pack_from_payload(payload: dict[str, Any], route_label: str | None = None) -> dict[str, Any]:
    routes = payload.get("routes") or []
    if route_label:
        for route in routes:
            if str(route.get("route_label") or "").upper() == route_label.upper():
                return route
        raise SystemExit(f"Route not found in pack: {route_label}")
    if len(routes) != 1:
        raise SystemExit("--route-label is required when the pack contains multiple routes")
    return routes[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-json", type=Path, required=True)
    parser.add_argument("--route-label")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--codex-bin", default="codex")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = read_json(args.pack_json)
    pack = route_pack_from_payload(payload, args.route_label)
    route_label = str(pack.get("route_label") or args.route_label or "route")
    output_json = args.output_json or DEFAULT_OUTPUT_DIR / f"{route_label}.review.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text(encoding="utf-8").rstrip()
    review_input = (
        f"{prompt}\n\n"
        "Review this route pack as a real outing, not as a certification object.\n\n"
        "```json\n"
        f"{json.dumps(pack, indent=2, sort_keys=False)}\n"
        "```\n"
    )
    command = [
        args.codex_bin,
        "exec",
        "--cd",
        str(REPO_ROOT),
        "--sandbox",
        "read-only",
        "--output-schema",
        str(args.schema),
        "-o",
        str(output_json),
        "-",
    ]
    process = subprocess.run(command, input=review_input, text=True, check=False)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
