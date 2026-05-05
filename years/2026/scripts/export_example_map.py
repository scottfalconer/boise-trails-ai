#!/usr/bin/env python3
"""Export a sanitized example copy of the generated outing menu map."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]

DEFAULT_INPUT_HTML = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map.html"
DEFAULT_OUTPUT_HTML = YEAR_DIR / "outputs" / "examples" / "2026-outing-menu-map.example.html"


def sanitize_map_html(html: str, repo_root: Path = REPO_ROOT) -> str:
    """Remove local absolute paths while preserving the interactive map payload."""

    sanitized = html.replace(str(repo_root) + "/", "")
    sanitized = sanitized.replace(str(repo_root), "")
    sanitized = re.sub(
        r'"(?:[^"]*/)?years/2026/outputs/private/([^"]+)"',
        r'"years/2026/outputs/example-redacted/\1"',
        sanitized,
    )
    sanitized = re.sub(
        r"`(?:[^`]+/)?years/2026/outputs/private/([^`]+)`",
        r"`years/2026/outputs/example-redacted/\1`",
        sanitized,
    )
    return sanitized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-html", type=Path, default=DEFAULT_INPUT_HTML)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    html = args.input_html.read_text(encoding="utf-8")
    sanitized = sanitize_map_html(html)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(sanitized, encoding="utf-8")
    print(f"Wrote {args.output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
