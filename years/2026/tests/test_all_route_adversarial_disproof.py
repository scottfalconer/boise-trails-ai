import importlib.util
import json
from pathlib import Path


YEAR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = YEAR_DIR.parent.parent
DISPROOF_JSON = YEAR_DIR / "checkpoints" / "all-route-adversarial-disproof-2026-05-16.json"
MAP_DATA_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_script(name: str):
    path = YEAR_DIR / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_route_candidates():
    map_data = read_json(MAP_DATA_JSON)
    rows = {}
    for package in map_data.get("packages") or []:
        for component in package.get("components") or []:
            label = str(
                component.get("field_menu_label")
                or component.get("label")
                or package.get("package_number")
                or ""
            )
            rows[label] = str(component.get("candidate_id") or "")
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

    assert artifact["summary"]["route_count"] == 43
    assert artifact["summary"]["proof_count"] == 43
    assert set(routes) == set(current)
    assert set(proofs) == set(current)
    assert {label: proof["candidate_id"] for label, proof in proofs.items()} == current
    assert artifact["summary"]["deterministic_same_credit_failure_count"] == 0


def test_h1_public_source_access_gap_is_explicitly_gated():
    artifact = read_json(DISPROOF_JSON)
    routes = {route["route_label"]: route for route in artifact["routes"]}
    proofs = {proof["labels"][0]: proof for proof in artifact["proofs"]}

    h1_route = routes["H1"]
    h1_proof = proofs["H1"]

    assert h1_route["decision"] == "HOLD_PUBLIC_ACCESS_RECHECK"
    assert h1_route["requires_field_walkthrough"] is True
    assert h1_route["public_source_status"] == "needs_public_access_confirmation"
    assert "Avimor owner page" in h1_route["bundle_boundary_global_attack"]

    assert h1_proof["status"] == "needs_public_access_confirmation"
    assert h1_proof["decision"] == "HOLD_PUBLIC_ACCESS_RECHECK"
    assert h1_proof["checks"]["public_owner_access_confirmation_present"] is False
    assert h1_proof["checks"]["access_confirmed_by_public_authoritative_source"] is False


def test_provisional_repairs_remain_explicitly_field_walkthrough_gated():
    artifact = read_json(DISPROOF_JSON)
    decisions = {route["route_label"]: route["decision"] for route in artifact["routes"]}

    assert decisions["FD03A"] == "HOLD_PROVISIONAL_FIELD_WALKTHROUGH"
    assert decisions["FD09A"] == "HOLD_PROVISIONAL_FIELD_WALKTHROUGH"
    assert decisions["FD14D"] == "HOLD_PROVISIONAL_FIELD_WALKTHROUGH"


def test_all_route_proofs_are_accepted_by_efficiency_and_repeat_audits():
    artifact = read_json(DISPROOF_JSON)
    current_candidate_ids = set(current_route_candidates().values())
    public_access_gated_candidate_ids = {
        str(proof["candidate_id"])
        for proof in artifact["proofs"]
        if proof.get("status") == "needs_public_access_confirmation"
    }
    expected_accepted_candidate_ids = current_candidate_ids - public_access_gated_candidate_ids

    efficiency_audit = load_script("route_efficiency_audit")
    repeat_audit = load_script("route_repeat_optimization_audit")

    efficiency_index = efficiency_audit.route_proof_index([artifact])
    repeat_index = repeat_audit.route_proof_index([artifact])

    assert expected_accepted_candidate_ids <= set(efficiency_index)
    assert expected_accepted_candidate_ids <= set(repeat_index)
    assert public_access_gated_candidate_ids.isdisjoint(set(efficiency_index))
    assert public_access_gated_candidate_ids.isdisjoint(set(repeat_index))


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
