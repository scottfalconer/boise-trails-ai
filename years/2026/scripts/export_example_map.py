#!/usr/bin/env python3
"""Export sanitized public copies of the generated outing menu artifacts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from block_day_packager import apply_human_route_names_to_map_data  # noqa: E402

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


PUBLIC_SAFE_PRIVATE_ANCHOR_LABEL_PATTERNS = {
    "1a-ms-04-1": "Full Sail Trailhead, N 36th St Parking",
    "fd14d-36th-street-chute-lower-36th": "Full Sail Trailhead, N 36th St Parking",
    "4c-ms-20-2": "Castle Rock-side prior parking anchor",
    "fd03a-chukar-butte-strava-anchor": "Chukar Butte prior parking anchor",
}

PUBLIC_SAFE_PRIVATE_ROUTE_PATTERNS = {
    "1a-ms-04-1",
    "fd14d-36th-street-chute-lower-36th",
}

PRIVATE_SOURCE_REPLACEMENTS = {
    "strava_activity_endpoint_cluster": "accepted_prior_activity_anchor",
    "strava_seen_prior_challenge_window": "accepted_prior_activity_anchor",
}


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
    match = re.search(r"const DATA = (.*?);\nconst map =", sanitized, flags=re.DOTALL)
    if match:
        data = sanitize_private_map_data(json.loads(match.group(1)))
        sanitized_payload = json.dumps(data, separators=(",", ":"))
        sanitized = sanitized[: match.start(1)] + sanitized_payload + sanitized[match.end(1) :]
    return sanitized


def extract_map_data_from_html(html: str) -> dict:
    match = re.search(r"const DATA = (.*?);\nconst map =", html, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not find embedded DATA payload in outing map HTML.")
    return json.loads(match.group(1))


def sanitize_map_data_json(html: str, repo_root: Path = REPO_ROOT) -> str:
    """Export the exact embedded map payload as sanitized shareable JSON."""

    data = sanitize_private_map_data(extract_map_data_from_html(html))
    sanitized = remove_local_paths(json.dumps(data, separators=(",", ":")), repo_root)
    sanitized = re.sub(
        r'"(?:[^"]*/)?years/2026/outputs/private/([^"]+)"',
        r'"years/2026/outputs/example-redacted/\1"',
        sanitized,
    )
    return sanitized + "\n"


def private_anchor_label(candidate_id: str, fallback: str = "Private prior parking anchor") -> str:
    for pattern, label in PUBLIC_SAFE_PRIVATE_ANCHOR_LABEL_PATTERNS.items():
        if pattern in candidate_id:
            return label
    return fallback


def is_public_safe_private_route(candidate_id: str) -> bool:
    return any(pattern in candidate_id for pattern in PUBLIC_SAFE_PRIVATE_ROUTE_PATTERNS)


def is_private_strava_trailhead(trailhead: dict) -> bool:
    text = " ".join(
        str(trailhead.get(key) or "")
        for key in ["name", "source", "parking_confidence"]
    ).lower()
    return "strava" in text


def sanitize_private_anchor_labels(text: str) -> str:
    replacements = {
        "Strava parking anchor 13": private_anchor_label("multi-start-1a-1a-ms-04-1"),
        "Strava parking anchor 21": private_anchor_label("multi-start-4c-4c-ms-20-2"),
        "Chukar Butte private Strava parking anchor": private_anchor_label(
            "accepted-replacement-fd03a-chukar-butte-strava-anchor-19"
        ),
    }
    for private_label, public_label in replacements.items():
        text = text.replace(private_label, public_label)
    return text


def sanitize_private_text_value(text: str) -> str:
    """Replace public-facing private evidence wording without changing route ids."""

    sanitized = sanitize_private_anchor_labels(text)
    replacements = {
        "private Strava-derived parking anchor": "accepted prior-parking anchor",
        "private Strava-derived anchor": "accepted prior-parking anchor",
        "Strava-derived parking anchor": "prior parking anchor",
        "strava_activity_endpoint_cluster": PRIVATE_SOURCE_REPLACEMENTS[
            "strava_activity_endpoint_cluster"
        ],
        "strava_seen_prior_challenge_window": PRIVATE_SOURCE_REPLACEMENTS[
            "strava_seen_prior_challenge_window"
        ],
    }
    for private_text, public_text in replacements.items():
        sanitized = sanitized.replace(private_text, public_text)
    return sanitized


def sanitize_private_text_fields(value):
    """Recursively sanitize private evidence labels in shareable map data."""

    if isinstance(value, dict):
        for key, child in list(value.items()):
            value[key] = sanitize_private_text_fields(child)
        return value
    if isinstance(value, list):
        for index, child in enumerate(value):
            value[index] = sanitize_private_text_fields(child)
        return value
    if isinstance(value, str):
        return sanitize_private_text_value(value)
    return value


def sanitize_private_map_data(data: dict) -> dict:
    """Remove private exact parking coordinates from public example data."""

    private_candidate_ids = {
        str(candidate_id)
        for candidate_id in (data.get("public_sanitization") or {}).get(
            "private_candidate_ids_redacted"
        )
        or []
        if not is_public_safe_private_route(str(candidate_id))
    }
    for candidate_id, cue in (data.get("route_cues") or {}).items():
        trailhead = cue.get("trailhead") or {}
        if is_private_strava_trailhead(trailhead) and not is_public_safe_private_route(str(candidate_id)):
            private_candidate_ids.add(str(candidate_id))
        elif is_private_strava_trailhead(trailhead):
            for source_key, public_value in PRIVATE_SOURCE_REPLACEMENTS.items():
                for key in ["source", "parking_confidence"]:
                    if trailhead.get(key) == source_key:
                        trailhead[key] = public_value

    for package in data.get("packages") or []:
        for component in package.get("components") or []:
            candidate_id = str(component.get("candidate_id") or "")
            if candidate_id in private_candidate_ids:
                component["trailhead"] = private_anchor_label(candidate_id)
        package["trailheads"] = sanitize_private_text_fields(package.get("trailheads") or [])

    for cue_candidate_id, cue in (data.get("route_cues") or {}).items():
        candidate_id = str(cue_candidate_id)
        if candidate_id not in private_candidate_ids:
            continue
        trailhead = cue.get("trailhead") or {}
        public_name = private_anchor_label(candidate_id)
        cue["trailhead"] = {
            key: value
            for key, value in trailhead.items()
            if key not in {"lat", "lon", "source", "parking_confidence"}
        }
        cue["trailhead"]["name"] = public_name
        cue["trailhead"]["privacy"] = "private_coordinates_redacted"

    collections = data.get("feature_collections") or {}
    for collection in collections.values():
        for feature in collection.get("features") or []:
            props = feature.get("properties") or {}
            candidate_id = str(props.get("candidate_id") or "")
            if str(props.get("trailhead") or "").lower().startswith("strava parking anchor"):
                props["trailhead"] = private_anchor_label(candidate_id)

    for collection_name in ["routes", "parking", "logistics"]:
        collection = collections.get(collection_name) or {}
        features = collection.get("features")
        if not isinstance(features, list):
            continue
        collection["features"] = [
            feature
            for feature in features
            if str((feature.get("properties") or {}).get("candidate_id") or "") not in private_candidate_ids
        ]
    data.setdefault("public_sanitization", {})["private_candidate_ids_redacted"] = sorted(private_candidate_ids)
    sanitize_private_text_fields(data)
    apply_human_route_names_to_map_data(data)
    return data


def remove_local_paths(text: str, repo_root: Path = REPO_ROOT) -> str:
    """Remove absolute repo paths from generated text artifacts."""

    sanitized = text.replace(str(repo_root) + "/", "")
    return sanitized.replace(str(repo_root), "")


def sanitize_menu_markdown(
    markdown: str, *, map_link: str, repo_root: Path = REPO_ROOT
) -> str:
    """Remove private paths from the written outing menu."""

    sanitized = sanitize_private_anchor_labels(remove_local_paths(markdown, repo_root))
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
