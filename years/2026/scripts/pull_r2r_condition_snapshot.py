#!/usr/bin/env python3
"""Snapshot R2R condition inputs used for planning freshness checks."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from personal_route_planner import DEFAULT_R2R_CONNECTOR_GEOJSON, read_json  # noqa: E402


DEFAULT_STATUSFY_URL = "https://statusfy.com/2082310001/11"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "inputs" / "open-data" / "dynamic-overlays-2026-05-06"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "r2r-condition-snapshot-2026-05-06.json"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def fetch_text(url: str, timeout: int = 20) -> str:
    request = Request(url, headers={"User-Agent": "boise-trails-ai-planner/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def visible_text(raw_html: str) -> str:
    text = re.sub(r"<script\b.*?</script>", " ", raw_html, flags=re.I | re.S)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_statusfy(raw_html: str, source_url: str) -> dict[str, Any]:
    text = visible_text(raw_html)
    status_detail = None
    match = re.search(r"Status Detail\s+(.+?)\s+Last Updated", text)
    if match:
        status_detail = match.group(1).strip()
    updated_age = None
    match = re.search(r"Updated\s+([^ ]+(?:\s+[^ ]+){0,3})\s+ago", text)
    if match:
        updated_age = match.group(1).strip() + " ago"
    epoch = None
    match = re.search(r"\b(17\d{8,})\b", text)
    if match:
        epoch = int(match.group(1))
    return {
        "source_url": source_url,
        "title": "Ridge to Rivers Trail Condition Reports",
        "status_detail": status_detail,
        "updated_age_text": updated_age,
        "updated_epoch": epoch,
        "raw_text_excerpt": text[:500],
    }


def normalize_r2r_conditions(r2r_geojson: Path) -> dict[str, Any]:
    data = read_json(r2r_geojson)
    rows = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        rows.append(
            {
                "trail_id": props.get("TrailID"),
                "trail_name": props.get("TrailName") or props.get("Name"),
                "trail_status": props.get("TrailStatus"),
                "condition": props.get("Condition"),
                "condition_date": props.get("ConditionDate"),
                "condition_notes": props.get("ConditionNotes"),
                "all_weather": props.get("AllWeather"),
                "special_management": props.get("SpecialManagement"),
                "special_management_comment": props.get("SpecialManagementComment"),
            }
        )
    return {
        "source_path": display_path(r2r_geojson),
        "feature_count": len(rows),
        "condition_counts": dict(sorted(Counter(row.get("condition") or "missing" for row in rows).items())),
        "trail_status_counts": dict(sorted(Counter(row.get("trail_status") or "missing" for row in rows).items())),
        "special_management_count": sum(1 for row in rows if row.get("special_management")),
        "rows": rows,
    }


def build_snapshot(r2r_geojson: Path, statusfy_url: str, *, fetch_statusfy: bool = True) -> dict[str, Any]:
    statusfy = None
    if fetch_statusfy:
        statusfy = parse_statusfy(fetch_text(statusfy_url), statusfy_url)
    return {
        "dataset": "r2r-condition-snapshot-2026-05-06",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_note": (
            "Statusfy gives the current general trail condition report; the R2R open-data file "
            "provides per-trail condition/status fields from the latest local pull."
        ),
        "statusfy": statusfy,
        "r2r_open_data_conditions": normalize_r2r_conditions(r2r_geojson),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2r-geojson", type=Path, default=DEFAULT_R2R_CONNECTOR_GEOJSON)
    parser.add_argument("--statusfy-url", default=DEFAULT_STATUSFY_URL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--no-fetch-statusfy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = build_snapshot(
        args.r2r_geojson,
        args.statusfy_url,
        fetch_statusfy=not args.no_fetch_statusfy,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(json.dumps({k: snapshot[k] for k in ["dataset", "generated_at_utc", "statusfy"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
