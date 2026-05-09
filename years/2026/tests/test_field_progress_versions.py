import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "field_progress_versions.py"


def load_module():
    spec = importlib.util.spec_from_file_location("field_progress_versions", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def field_tool_data():
    return {
        "schema": "boise_trails_field_tool_data_v1",
        "routes": [
            {"outing_id": "route-a", "segment_ids": ["101", "102"], "validation": {"passed": True}},
            {"outing_id": "route-b", "segment_ids": ["103"], "validation": {"passed": True}},
        ],
    }


def test_lock_original_and_apply_day_materializes_active_state_without_mutating_original(tmp_path):
    module = load_module()
    state_json = tmp_path / "state.private.json"
    ledger_json = tmp_path / "inputs" / "private" / "progress-ledger.json"
    version_root = tmp_path / "versions"
    state_json.write_text(
        json.dumps({"completed_segment_ids": [], "blocked_segment_ids": [], "pace_min_per_mile": 15.46}),
        encoding="utf-8",
    )
    review = {
        "schema": "boise_trails_activity_segment_review_v1",
        "planned_outing_id": "route-a",
        "completed_segment_ids": ["101", "102", "103"],
        "extra_completed_segment_ids": ["103"],
        "missed_segment_ids": [],
        "partial_segment_ids": [],
        "blocked_segment_ids": [],
    }

    lock = module.lock_original(
        epoch="pre-challenge-testing",
        state_json=state_json,
        version_root=version_root,
        ledger_json=ledger_json,
    )
    result = module.apply_day_review(
        epoch="pre-challenge-testing",
        day_id="2026-05-08-test-03",
        review=review,
        state_json=state_json,
        field_tool_data=field_tool_data(),
        version_root=version_root,
        ledger_json=ledger_json,
        run_reports=False,
        run_field_packet_export=False,
        copy_packet_artifacts=False,
    )

    active_state = json.loads(state_json.read_text(encoding="utf-8"))
    original_state = json.loads(Path(lock["state_snapshot_json"]).read_text(encoding="utf-8"))
    route_delta = json.loads(Path(result["route_delta_json"]).read_text(encoding="utf-8"))

    assert active_state["completed_segment_ids"] == [101, 102, 103]
    assert original_state["completed_segment_ids"] == []
    assert route_delta["completed_outing_ids"] == ["route-a", "route-b"]
    assert route_delta["outing_statuses"]["route-a"]["status"] == "completed_by_segments"
    assert Path(result["day_dir"]).name == "2026-05-08-test-03"


def test_reset_epoch_clears_ledger_and_locks_new_original(tmp_path):
    module = load_module()
    state_json = tmp_path / "state.private.json"
    ledger_json = tmp_path / "inputs" / "private" / "progress-ledger.json"
    version_root = tmp_path / "versions"
    state_json.write_text(
        json.dumps({"completed_segment_ids": [101], "blocked_segment_ids": [102], "blocked_trail_names": ["Closed"]}),
        encoding="utf-8",
    )
    ledger_json.parent.mkdir(parents=True)
    ledger_json.write_text(
        json.dumps(
            {
                "schema": "boise_trails_progress_ledger_v1",
                "events": [{"epoch": "pre-challenge-testing"}, {"epoch": "challenge-2026"}],
            }
        )
    )

    result = module.reset_epoch(
        epoch="challenge-2026",
        state_json=state_json,
        version_root=version_root,
        ledger_json=ledger_json,
        preserve_blocks=True,
    )

    active_state = json.loads(state_json.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_json.read_text(encoding="utf-8"))

    assert active_state["completed_segment_ids"] == []
    assert active_state["blocked_segment_ids"] == [102]
    assert active_state["blocked_trail_names"] == ["Closed"]
    assert ledger["events"] == [{"epoch": "pre-challenge-testing"}]
    assert Path(result["original"]["state_snapshot_json"]).exists()
