import importlib.util
import json
from pathlib import Path


YEAR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = YEAR_DIR.parent.parent
DISPROOF_JSON = YEAR_DIR / "checkpoints" / "all-route-adversarial-disproof-2026-05-16.json"
FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_script(name: str):
    path = YEAR_DIR / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_route_candidates():
    field_tool_data = read_json(FIELD_TOOL_DATA_JSON)
    rows = {}
    for route in field_tool_data.get("routes") or []:
        label = str(route.get("route_code") or route.get("label") or route.get("outing_id") or "")
        rows[label] = [str(candidate_id) for candidate_id in route.get("candidate_ids") or []]
    return rows


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def test_all_current_routes_have_adversarial_disproof_records():
    artifact = read_json(DISPROOF_JSON)
    current = current_route_candidates()

    routes = {route["route_label"]: route for route in artifact["routes"]}
    proofs = {proof["labels"][0]: proof for proof in artifact["proofs"]}

    # Every current route must have a record (no silent omissions), but the
    # registry no longer rubber-stamps them: each carries the real review state.
    assert artifact["summary"]["route_count"] == len(current)
    assert artifact["summary"]["proof_count"] == len(current)
    assert set(routes) == set(current)
    assert set(proofs) == set(current)
    assert {label: proof["candidate_ids"] for label, proof in proofs.items()} == current

    # The recorded failure count must equal the number of routes whose decision
    # is not the accepted decision (i.e. it is sourced, not hardcoded to 0).
    failed = [r for r in artifact["routes"] if r["decision"] != "HOLD_CURRENT_RECERTIFIED"]
    assert artifact["summary"]["deterministic_same_credit_failure_count"] == len(failed)
    assert set(artifact["summary"]["failed_route_labels"]) == {r["route_label"] for r in failed}
    # Accepted iff route_efficiency_achieved.
    assert artifact["summary"]["route_efficiency_achieved"] == (len(failed) == 0)


def test_all_route_disproof_registry_is_sourced_from_current_field_packet():
    artifact = read_json(DISPROOF_JSON)

    assert artifact["source_files"]["field_tool_data"] == "docs/field-packet/field-tool-data.json"
    assert artifact["source_files"]["route_review"] == (
        "years/2026/outputs/private/route-reviews/route-review-all-dev.review.json"
    )
    # Decisions are per-route now; they must sum to route_count and use only the
    # known vocabulary.
    decision_counts = artifact["summary"]["decision_counts"]
    assert sum(decision_counts.values()) == artifact["summary"]["route_count"]
    assert set(decision_counts) <= {"HOLD_CURRENT_RECERTIFIED", "NEEDS_REANCHOR_OR_WAIVER"}


def test_failed_routes_are_excluded_from_efficiency_and_repeat_proof_indexes():
    """Fail-closed contract: only accepted routes' candidate_ids may be indexed
    as proven; routes the dominance gate failed must drop out so their
    optimization warnings re-open."""
    artifact = read_json(DISPROOF_JSON)
    routes_by_label = {route["route_label"]: route for route in artifact["routes"]}
    current = current_route_candidates()

    accepted_candidate_ids = set()
    failed_candidate_ids = set()
    for label, candidate_ids in current.items():
        bucket = (
            accepted_candidate_ids
            if routes_by_label[label]["decision"] == "HOLD_CURRENT_RECERTIFIED"
            else failed_candidate_ids
        )
        bucket.update(candidate_ids)
    failed_only = failed_candidate_ids - accepted_candidate_ids

    efficiency_audit = load_script("route_efficiency_audit")
    repeat_audit = load_script("route_repeat_optimization_audit")
    efficiency_index = set(efficiency_audit.route_proof_index([artifact]))
    repeat_index = set(repeat_audit.route_proof_index([artifact]))

    assert accepted_candidate_ids <= efficiency_index
    assert accepted_candidate_ids <= repeat_index
    assert failed_only.isdisjoint(efficiency_index)
    assert failed_only.isdisjoint(repeat_index)


def test_all_route_disproof_artifact_stays_public_safe():
    artifact = read_json(DISPROOF_JSON)
    text = DISPROOF_JSON.read_text(encoding="utf-8")
    forbidden_keys = {
        "lat",
        "latitude",
        "lon",
        "lng",
        "longitude",
        "coordinate",
        "coordinates",
        "activity_id",
        "activity_ids",
        "raw_activity_id",
        "raw_activity_ids",
        "athlete_id",
        "token",
    }

    assert forbidden_keys.isdisjoint({key.lower() for key in walk_keys(artifact)})
    assert "private Strava" not in text
    assert "<gpx" not in text.lower()
