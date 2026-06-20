#!/usr/bin/env python3
"""Manage segment-first BTC progress ledger and private route-state versions."""

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
sys.path.insert(0, str(SCRIPT_DIR))

import field_progress_report  # noqa: E402


DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_LEDGER_JSON = YEAR_DIR / "inputs" / "personal" / "private" / "progress-ledger.json"
DEFAULT_VERSION_ROOT = YEAR_DIR / "outputs" / "private" / "progress" / "versions"
DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_OFFICIAL_GEOJSON = YEAR_DIR / "inputs" / "official" / "api-pull-2026-06-13" / "official_foot_segments.geojson"
DEFAULT_PACKET_DIR = REPO_ROOT / "docs" / "field-packet"
SNAPSHOT_PACKET_FILES = ["index.html", "live-map.html", "manifest.json", "field-tool-data.json", "service-worker.js"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_ids(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    return sorted({str(value) for value in values or [] if value is not None}, key=lambda item: (len(item), item))


def state_ids(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[int | str]:
    result = []
    for value in normalized_ids(values):
        try:
            result.append(int(value))
        except ValueError:
            result.append(value)
    return sorted(result, key=lambda item: (isinstance(item, str), item))


def empty_ledger() -> dict[str, Any]:
    return {"schema": "boise_trails_progress_ledger_v1", "events": []}


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_ledger()
    ledger = read_json(path)
    ledger.setdefault("schema", "boise_trails_progress_ledger_v1")
    ledger.setdefault("events", [])
    return ledger


def original_dir(version_root: Path, epoch: str) -> Path:
    return version_root / epoch / "original"


def day_dir(version_root: Path, epoch: str, day_id: str) -> Path:
    return version_root / epoch / "days" / day_id


def lock_original(
    *,
    epoch: str,
    state_json: Path = DEFAULT_STATE_JSON,
    version_root: Path = DEFAULT_VERSION_ROOT,
    ledger_json: Path = DEFAULT_LEDGER_JSON,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = original_dir(version_root, epoch)
    state_snapshot = out_dir / "state.original.json"
    metadata_path = out_dir / "metadata.json"
    if state_snapshot.exists() and metadata_path.exists() and not force:
        return read_json(metadata_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(state_json, state_snapshot)
    ledger_snapshot = out_dir / "progress-ledger-at-lock.json"
    write_json(ledger_snapshot, load_ledger(ledger_json))
    metadata = {
        "schema": "boise_trails_progress_original_lock_v1",
        "epoch": epoch,
        "locked_at": now_iso(),
        "state_snapshot_json": str(state_snapshot),
        "ledger_snapshot_json": str(ledger_snapshot),
        "state_source_json": str(state_json),
    }
    write_json(metadata_path, metadata)
    return metadata


def review_event(epoch: str, day_id: str, review: dict[str, Any]) -> dict[str, Any]:
    completed = set(normalized_ids(review.get("completed_segment_ids")))
    completed.update(normalized_ids(review.get("extra_completed_segment_ids")))
    return {
        "schema": "boise_trails_progress_event_v1",
        "epoch": epoch,
        "day_id": day_id,
        "recorded_at": now_iso(),
        "planned_outing_id": review.get("planned_outing_id"),
        "evidence_refs": review.get("evidence_refs") or [],
        "completed_segment_ids": normalized_ids(completed),
        "planned_completed_segment_ids": normalized_ids(review.get("planned_completed_segment_ids")),
        "extra_completed_segment_ids": normalized_ids(review.get("extra_completed_segment_ids")),
        "missed_segment_ids": normalized_ids(review.get("missed_segment_ids")),
        "partial_segment_ids": normalized_ids(review.get("partial_segment_ids")),
        "near_touch_segment_ids": normalized_ids(review.get("near_touch_segment_ids")),
        "blocked_segment_ids": normalized_ids(review.get("blocked_segment_ids")),
        "blocked_trail_names": sorted(str(item) for item in review.get("blocked_trail_names") or []),
        "notes": review.get("notes") or "",
    }


def segment_state_from_ledger(ledger: dict[str, Any], epoch: str) -> dict[str, list[str]]:
    completed: set[str] = set()
    missed: set[str] = set()
    partial: set[str] = set()
    blocked: set[str] = set()
    blocked_trails: set[str] = set()
    for event in ledger.get("events") or []:
        if event.get("epoch") != epoch:
            continue
        completed.update(normalized_ids(event.get("completed_segment_ids")))
        completed.update(normalized_ids(event.get("extra_completed_segment_ids")))
        missed.update(normalized_ids(event.get("missed_segment_ids")))
        partial.update(normalized_ids(event.get("partial_segment_ids")))
        blocked.update(normalized_ids(event.get("blocked_segment_ids")))
        blocked_trails.update(str(item) for item in event.get("blocked_trail_names") or [])
    completed.difference_update(blocked)
    return {
        "completed_segment_ids": normalized_ids(completed),
        "missed_segment_ids": normalized_ids(missed),
        "partial_segment_ids": normalized_ids(partial),
        "blocked_segment_ids": normalized_ids(blocked),
        "blocked_trail_names": sorted(blocked_trails),
    }


def materialize_state(base_state: dict[str, Any], segment_state: dict[str, list[str]]) -> dict[str, Any]:
    state = dict(base_state)
    state["completed_segment_ids"] = state_ids(segment_state.get("completed_segment_ids"))
    state["blocked_segment_ids"] = state_ids(segment_state.get("blocked_segment_ids"))
    state["blocked_trail_names"] = list(segment_state.get("blocked_trail_names") or [])
    return state


def route_delta(field_tool_data: dict[str, Any], segment_state: dict[str, list[str]]) -> dict[str, Any]:
    routes_by_id = field_progress_report.route_index(field_tool_data)
    completed = set(segment_state.get("completed_segment_ids") or [])
    blocked = set(segment_state.get("blocked_segment_ids") or [])
    statuses = field_progress_report.outing_statuses_from_segments(routes_by_id, completed, blocked)
    completed_outing_ids = [
        outing_id for outing_id, status in statuses.items() if status["status"] == "completed_by_segments"
    ]
    inactive_outing_ids = [
        outing_id
        for outing_id, status in statuses.items()
        if status["status"] == "inactive_no_remaining_new_credit"
    ]
    return {
        "schema": "boise_trails_route_delta_v1",
        "completed_outing_ids": field_progress_report.normalized_ids(completed_outing_ids),
        "inactive_outing_ids": field_progress_report.normalized_ids(inactive_outing_ids),
        "outing_statuses": statuses,
    }


def progress_input_from_segment_state(segment_state: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "schema": "boise_trails_segment_progress_input_v1",
        "completed_segment_ids": state_ids(segment_state.get("completed_segment_ids")),
        "missed_segment_ids": state_ids(segment_state.get("missed_segment_ids")),
        "partial_segment_ids": state_ids(segment_state.get("partial_segment_ids")),
        "blocked_segment_ids": state_ids(segment_state.get("blocked_segment_ids")),
        "blocked_trail_names": list(segment_state.get("blocked_trail_names") or []),
    }


def copy_packet_snapshot(packet_dir: Path, destination: Path) -> list[str]:
    copied = []
    target = destination / "field-packet"
    target.mkdir(parents=True, exist_ok=True)
    for name in SNAPSHOT_PACKET_FILES:
        source = packet_dir / name
        if source.exists():
            dest = target / name
            shutil.copy2(source, dest)
            copied.append(str(dest))
    zip_path = packet_dir / "gpx" / "all-field-packet-gpx.zip"
    if zip_path.exists():
        dest = target / "all-field-packet-gpx.zip"
        shutil.copy2(zip_path, dest)
        copied.append(str(dest))
    return copied


def run_snapshot_reports(
    *,
    progress_input_json: Path,
    day_output_dir: Path,
    field_tool_data_json: Path,
    official_geojson: Path,
) -> dict[str, str]:
    progress_json = day_output_dir / "field-progress.json"
    progress_md = day_output_dir / "field-progress.md"
    recert_json = day_output_dir / "field-recertification.json"
    recert_md = day_output_dir / "field-recertification.md"
    commands = [
        [
            sys.executable,
            str(SCRIPT_DIR / "field_progress_report.py"),
            "--field-tool-data-json",
            str(field_tool_data_json),
            "--official-geojson",
            str(official_geojson),
            "--progress-json",
            str(progress_input_json),
            "--output-json",
            str(progress_json),
            "--output-md",
            str(progress_md),
        ],
        [
            sys.executable,
            str(SCRIPT_DIR / "field_recertification_report.py"),
            "--field-tool-data-json",
            str(field_tool_data_json),
            "--official-geojson",
            str(official_geojson),
            "--progress-json",
            str(progress_input_json),
            "--output-json",
            str(recert_json),
            "--output-md",
            str(recert_md),
            "--skip-heavy-optimizer",
        ],
    ]
    results = []
    for command in commands:
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
        results.append(
            {
                "command": " ".join(command),
                "stdout_tail": completed.stdout.strip().splitlines()[-5:],
                "stderr_tail": completed.stderr.strip().splitlines()[-5:],
            }
        )
    write_json(day_output_dir / "report-commands.json", {"commands": results})
    return {
        "progress_report_json": str(progress_json),
        "progress_report_md": str(progress_md),
        "recertification_report_json": str(recert_json),
        "recertification_report_md": str(recert_md),
    }


def run_packet_export(progress_input_json: Path, day_output_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "export_mobile_field_packet.py"),
        "--progress-json",
        str(progress_input_json),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    result = {
        "command": " ".join(command),
        "stdout_tail": completed.stdout.strip().splitlines()[-8:],
        "stderr_tail": completed.stderr.strip().splitlines()[-8:],
    }
    write_json(day_output_dir / "field-packet-export-command.json", result)
    return result


def apply_day_review(
    *,
    epoch: str,
    day_id: str,
    review: dict[str, Any],
    state_json: Path = DEFAULT_STATE_JSON,
    field_tool_data: dict[str, Any] | None = None,
    field_tool_data_json: Path = DEFAULT_FIELD_TOOL_DATA_JSON,
    official_geojson: Path = DEFAULT_OFFICIAL_GEOJSON,
    version_root: Path = DEFAULT_VERSION_ROOT,
    ledger_json: Path = DEFAULT_LEDGER_JSON,
    packet_dir: Path = DEFAULT_PACKET_DIR,
    run_reports: bool = True,
    run_field_packet_export: bool = True,
    copy_packet_artifacts: bool = True,
) -> dict[str, Any]:
    original = lock_original(epoch=epoch, state_json=state_json, version_root=version_root, ledger_json=ledger_json)
    ledger = load_ledger(ledger_json)
    ledger["events"] = [event for event in ledger.get("events") or [] if not (event.get("epoch") == epoch and event.get("day_id") == day_id)]
    ledger["events"].append(review_event(epoch, day_id, review))
    write_json(ledger_json, ledger)
    segment_state = segment_state_from_ledger(ledger, epoch)
    base_state = read_json(Path(original["state_snapshot_json"]))
    active_state = materialize_state(base_state, segment_state)
    write_json(state_json, active_state)

    out_dir = day_dir(version_root, epoch, day_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_json = out_dir / "activity-review.json"
    progress_json = out_dir / "progress-input.json"
    state_patch_json = out_dir / "state-patch.json"
    route_delta_json = out_dir / "route-delta.json"
    write_json(review_json, review)
    write_json(progress_json, progress_input_from_segment_state(segment_state))
    write_json(
        state_patch_json,
        {
            "completed_segment_ids": active_state.get("completed_segment_ids") or [],
            "blocked_segment_ids": active_state.get("blocked_segment_ids") or [],
            "blocked_trail_names": active_state.get("blocked_trail_names") or [],
            "_source": "field_progress_versions.py",
        },
    )
    field_data = field_tool_data if field_tool_data is not None else read_json(field_tool_data_json)
    delta = route_delta(field_data, segment_state)
    write_json(route_delta_json, delta)
    report_paths = (
        run_snapshot_reports(
            progress_input_json=progress_json,
            day_output_dir=out_dir,
            field_tool_data_json=field_tool_data_json,
            official_geojson=official_geojson,
        )
        if run_reports
        else {}
    )
    packet_export = run_packet_export(progress_json, out_dir) if run_field_packet_export else None
    copied_packet = copy_packet_snapshot(packet_dir, out_dir) if copy_packet_artifacts else []
    result = {
        "schema": "boise_trails_progress_day_apply_v1",
        "epoch": epoch,
        "day_id": day_id,
        "day_dir": str(out_dir),
        "original": original,
        "ledger_json": str(ledger_json),
        "state_json": str(state_json),
        "activity_review_json": str(review_json),
        "progress_input_json": str(progress_json),
        "state_patch_json": str(state_patch_json),
        "route_delta_json": str(route_delta_json),
        "field_packet_export": packet_export,
        "copied_packet_artifacts": copied_packet,
        **report_paths,
    }
    write_json(out_dir / "apply-result.json", result)
    return result


def reset_epoch(
    *,
    epoch: str,
    state_json: Path = DEFAULT_STATE_JSON,
    version_root: Path = DEFAULT_VERSION_ROOT,
    ledger_json: Path = DEFAULT_LEDGER_JSON,
    preserve_blocks: bool = False,
    force_lock: bool = True,
) -> dict[str, Any]:
    state = read_json(state_json)
    reset_state = dict(state)
    reset_state["completed_segment_ids"] = []
    if not preserve_blocks:
        reset_state["blocked_segment_ids"] = []
        reset_state["blocked_trail_names"] = []
    write_json(state_json, reset_state)
    ledger = load_ledger(ledger_json)
    ledger["events"] = [event for event in ledger.get("events") or [] if event.get("epoch") != epoch]
    write_json(ledger_json, ledger)
    original = lock_original(
        epoch=epoch,
        state_json=state_json,
        version_root=version_root,
        ledger_json=ledger_json,
        force=force_lock,
    )
    result = {
        "schema": "boise_trails_progress_epoch_reset_v1",
        "epoch": epoch,
        "reset_at": now_iso(),
        "preserve_blocks": preserve_blocks,
        "state_json": str(state_json),
        "ledger_json": str(ledger_json),
        "original": original,
    }
    write_json(version_root / epoch / "reset-latest.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock-original")
    lock.add_argument("--epoch", required=True)
    lock.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    lock.add_argument("--ledger-json", type=Path, default=DEFAULT_LEDGER_JSON)
    lock.add_argument("--version-root", type=Path, default=DEFAULT_VERSION_ROOT)
    lock.add_argument("--force", action="store_true")

    apply = sub.add_parser("apply-day")
    apply.add_argument("--epoch", required=True)
    apply.add_argument("--day-id", required=True)
    apply.add_argument("--review-json", type=Path, required=True)
    apply.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    apply.add_argument("--ledger-json", type=Path, default=DEFAULT_LEDGER_JSON)
    apply.add_argument("--version-root", type=Path, default=DEFAULT_VERSION_ROOT)
    apply.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    apply.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    apply.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    apply.add_argument("--skip-reports", action="store_true")
    apply.add_argument("--skip-packet-export", action="store_true")
    apply.add_argument("--skip-packet-copy", action="store_true")

    reset = sub.add_parser("reset-epoch")
    reset.add_argument("--epoch", required=True)
    reset.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    reset.add_argument("--ledger-json", type=Path, default=DEFAULT_LEDGER_JSON)
    reset.add_argument("--version-root", type=Path, default=DEFAULT_VERSION_ROOT)
    block_mode = reset.add_mutually_exclusive_group()
    block_mode.add_argument("--preserve-blocks", action="store_true")
    block_mode.add_argument("--clear-blocks", action="store_true")
    reset.add_argument("--no-force-lock", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "lock-original":
        result = lock_original(
            epoch=args.epoch,
            state_json=args.state_json,
            version_root=args.version_root,
            ledger_json=args.ledger_json,
            force=args.force,
        )
    elif args.command == "apply-day":
        result = apply_day_review(
            epoch=args.epoch,
            day_id=args.day_id,
            review=read_json(args.review_json),
            state_json=args.state_json,
            version_root=args.version_root,
            ledger_json=args.ledger_json,
            field_tool_data_json=args.field_tool_data_json,
            official_geojson=args.official_geojson,
            packet_dir=args.packet_dir,
            run_reports=not args.skip_reports,
            run_field_packet_export=not args.skip_packet_export,
            copy_packet_artifacts=not args.skip_packet_copy,
        )
    else:
        result = reset_epoch(
            epoch=args.epoch,
            state_json=args.state_json,
            version_root=args.version_root,
            ledger_json=args.ledger_json,
            preserve_blocks=args.preserve_blocks,
            force_lock=not args.no_force_lock,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
