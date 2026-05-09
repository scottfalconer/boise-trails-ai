#!/usr/bin/env python3
"""Reset the private 2026 planner state to a clean challenge-start map."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]

DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_RESET_RECORD = YEAR_DIR / "outputs" / "private" / "reset" / "challenge-start-reset-latest.json"
DEFAULT_MAP_DATA_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_MAP_HTML = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map.html"
DEFAULT_OUTING_MENU_MD = YEAR_DIR / "outputs" / "private" / "2026-outing-menu.md"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def reset_state_fields(state: dict[str, Any], preserve_blocks: bool = False) -> dict[str, Any]:
    """Return state with challenge progress reset and personal settings preserved."""

    reset = dict(state)
    reset["completed_segment_ids"] = []
    if not preserve_blocks:
        reset["blocked_segment_ids"] = []
        reset["blocked_trail_names"] = []
    else:
        reset["blocked_segment_ids"] = list(state.get("blocked_segment_ids") or [])
        reset["blocked_trail_names"] = list(state.get("blocked_trail_names") or [])
    return reset


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_state(state_json: Path, backup_dir: Path, stamp: str) -> Path | None:
    if not state_json.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{state_json.stem}-{stamp}.json"
    shutil.copy2(state_json, backup_path)
    return backup_path


def build_pipeline_commands(state_json: Path) -> list[list[str]]:
    py = sys.executable
    return [
        [
            py,
            "years/2026/scripts/personal_route_planner.py",
            "--state",
            str(state_json),
            "--output-json",
            "years/2026/outputs/private/personal-route-menu.json",
            "--output-md",
            "years/2026/outputs/private/personal-route-menu.md",
        ],
        [py, "years/2026/scripts/block_route_candidate_pass.py"],
        [py, "years/2026/scripts/block_day_packager.py"],
        [py, "years/2026/scripts/block_combo_route_pass.py"],
        [
            py,
            "years/2026/scripts/block_day_packager.py",
            "--route-pass-json",
            "years/2026/outputs/private/route-blocks/block-combo-route-pass-v1.json",
            "--basename",
            "block-combo-day-package-pass-v1",
        ],
        [py, "years/2026/scripts/block_route_assembler.py"],
        [py, "years/2026/scripts/block_hybrid_route_pass.py"],
        [
            py,
            "years/2026/scripts/block_day_packager.py",
            "--route-pass-json",
            "years/2026/outputs/private/route-blocks/block-hybrid-route-pass-v1.json",
            "--basename",
            "block-hybrid-day-package-pass-v1",
        ],
        [py, "years/2026/scripts/manual_route_design_pass.py"],
        [py, "years/2026/scripts/human_loop_plan.py"],
    ]


def run_pipeline(commands: list[list[str]]) -> list[dict[str, Any]]:
    results = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        results.append(
            {
                "command": " ".join(command),
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout.strip().splitlines()[-5:],
                "stderr_tail": completed.stderr.strip().splitlines()[-5:],
            }
        )
    return results


def verify_reset_outputs(map_data_json: Path) -> dict[str, Any]:
    map_data = read_json(map_data_json)
    progress = map_data.get("progress") or {}
    completed = list(progress.get("completed_segment_ids") or [])
    blocked = list(progress.get("blocked_segment_ids") or [])
    return {
        "map_data_json": str(map_data_json),
        "progress": {
            "completed_segment_ids": completed,
            "blocked_segment_ids": blocked,
        },
        "clean_start": completed == [] and blocked == [],
        "summary": map_data.get("summary") or {},
        "open_outing_source_package_count": len(map_data.get("packages") or []),
        "route_cue_count": len(map_data.get("route_cues") or {}),
        "map_rendered_passed": (map_data.get("map_validation") or {}).get("rendered_passed"),
    }


def build_reset_record(
    *,
    reset_at: str,
    state_json: Path,
    backup_path: Path | None,
    preserve_blocks: bool,
    pipeline_results: list[dict[str, Any]],
    output_verification: dict[str, Any],
    locked_original: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "reset_at_utc": reset_at,
        "purpose": "Clean challenge-start state for testing and future event-day reset.",
        "state_json": str(state_json),
        "state_backup_json": str(backup_path) if backup_path else None,
        "progress_reset": {
            "completed_segment_ids": [],
            "blocked_segment_ids": "preserved" if preserve_blocks else [],
            "blocked_trail_names": "preserved" if preserve_blocks else [],
        },
        "pipeline": pipeline_results,
        "outputs": {
            "canonical_map_html": str(DEFAULT_MAP_HTML),
            "canonical_outing_menu_md": str(DEFAULT_OUTING_MENU_MD),
            "map_data_json": str(DEFAULT_MAP_DATA_JSON),
        },
        "locked_original": locked_original,
        "verification": output_verification,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--reset-record-json", type=Path, default=DEFAULT_RESET_RECORD)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument(
        "--preserve-blocks",
        action="store_true",
        help="Keep currently configured blocked segments/trails while clearing completed progress.",
    )
    parser.add_argument("--skip-pipeline", action="store_true", help="Only reset state and write the reset record.")
    parser.add_argument("--dry-run", action="store_true", help="Print the reset action without writing files.")
    parser.add_argument("--lock-original-epoch", help="After resetting state, lock this epoch's original progress baseline.")
    parser.add_argument("--force-lock-original", action="store_true", help="Replace an existing original epoch snapshot.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reset_at = timestamp()
    state = read_json(args.state_json)
    reset_state = reset_state_fields(state, preserve_blocks=args.preserve_blocks)
    commands = build_pipeline_commands(args.state_json)

    if args.dry_run:
        print(f"Would reset {args.state_json}")
        print(json.dumps({key: reset_state.get(key) for key in ["completed_segment_ids", "blocked_segment_ids", "blocked_trail_names"]}, indent=2))
        if not args.skip_pipeline:
            print("Would run:")
            for command in commands:
                print("  " + " ".join(command))
        return 0

    backup_path = backup_state(
        args.state_json,
        YEAR_DIR / "outputs" / "private" / "reset" / "state-backups",
        reset_at,
    )
    write_json(args.state_json, reset_state)
    locked_original = None
    if args.lock_original_epoch:
        import field_progress_versions

        locked_original = field_progress_versions.lock_original(
            epoch=args.lock_original_epoch,
            state_json=args.state_json,
            force=args.force_lock_original,
        )
    pipeline_results = [] if args.skip_pipeline else run_pipeline(commands)
    output_verification = (
        verify_reset_outputs(args.map_data_json)
        if args.map_data_json.exists() and not args.skip_pipeline
        else {"clean_start": args.skip_pipeline, "map_data_json": str(args.map_data_json)}
    )
    record = build_reset_record(
        reset_at=reset_at,
        state_json=args.state_json,
        backup_path=backup_path,
        preserve_blocks=args.preserve_blocks,
        pipeline_results=pipeline_results,
        output_verification=output_verification,
        locked_original=locked_original,
    )
    write_json(args.reset_record_json, record)
    print(f"Reset state: {args.state_json}")
    print(f"State backup: {backup_path}")
    print(f"Reset record: {args.reset_record_json}")
    print(f"Clean map progress: {output_verification.get('clean_start')}")
    return 0 if output_verification.get("clean_start") else 1


if __name__ == "__main__":
    raise SystemExit(main())
