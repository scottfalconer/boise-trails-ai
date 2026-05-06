#!/usr/bin/env python3
"""Export sanitized public copies of the generated outing menu artifacts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]

DEFAULT_INPUT_HTML = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map.html"
DEFAULT_OUTPUT_HTML = YEAR_DIR / "outputs" / "examples" / "2026-outing-menu-map.example.html"
DEFAULT_ROOT_OUTPUT_HTML = REPO_ROOT / "outing-menu-map.html"
DEFAULT_OUTPUT_DATA_JSON = YEAR_DIR / "outputs" / "examples" / "2026-outing-menu-map-data.example.json"
DEFAULT_ROOT_OUTPUT_DATA_JSON = REPO_ROOT / "outing-menu-map-data.json"
DEFAULT_INPUT_MD = YEAR_DIR / "outputs" / "private" / "2026-outing-menu.md"
DEFAULT_OUTPUT_MD = YEAR_DIR / "outputs" / "examples" / "2026-outing-menu.example.md"
DEFAULT_ROOT_OUTPUT_MD = REPO_ROOT / "outing-menu.md"
DEFAULT_INPUT_SCREENSHOT = YEAR_DIR / "outputs" / "private" / "outing-menu-map-door-to-door.png"
DEFAULT_ROOT_OUTPUT_SCREENSHOT = REPO_ROOT / "outing-menu-map.png"


def sanitize_map_html(html: str, repo_root: Path = REPO_ROOT) -> str:
    """Remove local absolute paths while preserving the interactive map payload."""

    sanitized = remove_local_paths(html, repo_root)
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


def extract_map_data_from_html(html: str) -> dict:
    match = re.search(r"const DATA = (.*?);\nconst map =", html, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not find embedded DATA payload in outing map HTML.")
    return json.loads(match.group(1))


def sanitize_map_data_json(html: str, repo_root: Path = REPO_ROOT) -> str:
    """Export the exact embedded map payload as sanitized shareable JSON."""

    data = extract_map_data_from_html(sanitize_map_html(html, repo_root=repo_root))
    return json.dumps(data, separators=(",", ":")) + "\n"


def remove_local_paths(text: str, repo_root: Path = REPO_ROOT) -> str:
    """Remove absolute repo paths from generated text artifacts."""

    sanitized = text.replace(str(repo_root) + "/", "")
    return sanitized.replace(str(repo_root), "")


def sanitize_menu_markdown(
    markdown: str, *, map_link: str, repo_root: Path = REPO_ROOT
) -> str:
    """Remove private paths from the written outing menu."""

    sanitized = remove_local_paths(markdown, repo_root)
    sanitized = re.sub(
        r"- Map: `[^`]*2026-outing-menu-map\.html`",
        f"- Map: `{map_link}`",
        sanitized,
    )
    sanitized = re.sub(
        r"`(?:[^`]+/)?years/2026/outputs/private/([^`]+)`",
        r"`years/2026/outputs/example-redacted/\1`",
        sanitized,
    )
    return sanitized


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-html", type=Path, default=DEFAULT_INPUT_HTML)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--root-output-html", type=Path, default=DEFAULT_ROOT_OUTPUT_HTML)
    parser.add_argument("--output-data-json", type=Path, default=DEFAULT_OUTPUT_DATA_JSON)
    parser.add_argument("--root-output-data-json", type=Path, default=DEFAULT_ROOT_OUTPUT_DATA_JSON)
    parser.add_argument("--input-md", type=Path, default=DEFAULT_INPUT_MD)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--root-output-md", type=Path, default=DEFAULT_ROOT_OUTPUT_MD)
    parser.add_argument("--input-screenshot", type=Path, default=DEFAULT_INPUT_SCREENSHOT)
    parser.add_argument(
        "--root-output-screenshot",
        type=Path,
        default=DEFAULT_ROOT_OUTPUT_SCREENSHOT,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    html = args.input_html.read_text(encoding="utf-8")
    sanitized_html = sanitize_map_html(html)
    write_text(args.output_html, sanitized_html)
    write_text(args.root_output_html, sanitized_html)
    sanitized_data_json = sanitize_map_data_json(html)
    write_text(args.output_data_json, sanitized_data_json)
    write_text(args.root_output_data_json, sanitized_data_json)

    markdown = args.input_md.read_text(encoding="utf-8")
    write_text(
        args.output_md,
        sanitize_menu_markdown(markdown, map_link="2026-outing-menu-map.example.html"),
    )
    write_text(
        args.root_output_md,
        sanitize_menu_markdown(markdown, map_link="outing-menu-map.html"),
    )

    if args.input_screenshot.exists():
        args.root_output_screenshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.input_screenshot, args.root_output_screenshot)
        print(f"Wrote {args.root_output_screenshot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
