import json
from pathlib import Path


YEAR_DIR = Path(__file__).resolve().parents[1]
PUBLIC_REEVALUATION_JSON = YEAR_DIR / "checkpoints" / "public-source-route-reevaluation-2026-05-16.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def test_public_source_reevaluation_records_h1_user_confirmed_access():
    artifact = read_json(PUBLIC_REEVALUATION_JSON)
    impacts = {impact["route_label"]: impact for impact in artifact["route_impacts"]}

    assert artifact["frame_decision"] == "hold"
    assert artifact["summary"]["route_set_mutation_required_now"] is False
    assert artifact["summary"]["routes_downgraded"] == 0
    assert artifact["summary"]["routes_downgraded_by_public_source_only"] == 1
    assert artifact["summary"]["route_labels_requiring_new_access_proof"] == []
    assert artifact["summary"]["route_labels_with_user_confirmed_access"] == ["H1"]
    assert artifact["resolution"]["status"] == "resolved_for_private_field_use"

    h1 = impacts["H1"]
    assert h1["status_before"] == "public_access_recheck"
    assert h1["status_after"] == "accepted_current_user_confirmed"
    assert h1["decision_after"] == "HOLD_PROVEN_CURRENT"
    assert h1["user_confirmed_access"] is True
    assert h1["access_status"] == "accepted_user_reviewed"
    assert "OSM plus AllTrails" in h1["why_changed"]
    assert "No known accepted same-credit" in h1["what_did_not_change"]


def test_public_source_reevaluation_keeps_existing_bogus_condition_gate():
    artifact = read_json(PUBLIC_REEVALUATION_JSON)
    impacts = {impact["route_label"]: impact for impact in artifact["route_impacts"]}

    bogus = impacts["Bogus routes"]
    assert bogus["status_after"] == "condition_gated_reaffirmed"
    assert bogus["decision_after"] == "HOLD_CONDITION_GATED"
    assert "June 19, 2026" in bogus["why_changed"]


def test_public_source_reevaluation_artifact_stays_public_safe():
    artifact = read_json(PUBLIC_REEVALUATION_JSON)
    text = PUBLIC_REEVALUATION_JSON.read_text(encoding="utf-8")
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
