import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "derive_strava_parking_anchors.py"


def load_module():
    spec = importlib.util.spec_from_file_location("derive_strava_parking_anchors", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def activity(activity_id, local_date, start, end, distance=5000, sport="Run"):
    return {
        "id": activity_id,
        "name": f"Run {activity_id}",
        "sport_type": sport,
        "type": sport,
        "distance": distance,
        "start_date_local": f"{local_date}T07:00:00Z",
        "start_latlng": start,
        "end_latlng": end,
    }


def test_candidate_endpoint_records_use_pull_summary_windows_and_exclude_home():
    module = load_module()
    windows = [
        {"label": "2025_challenge_window", "start": module.date(2025, 6, 19), "end": module.date(2025, 7, 19)}
    ]
    home = (-116.2, 43.6)
    records, counters = module.candidate_endpoint_records(
        [
            activity(1, "2025-06-20", [43.61, -116.21], [43.62, -116.22]),
            activity(2, "2025-06-20", [43.6001, -116.2001], [43.63, -116.23]),
            activity(3, "2025-08-20", [43.64, -116.24], [43.65, -116.25]),
            activity(4, "2025-06-20", [43.66, -116.26], [43.67, -116.27], sport="Ride"),
        ],
        windows,
        home_point=home,
        exclude_home_radius_miles=0.25,
    )

    assert [record["activity_id"] for record in records] == [1, 1, 2]
    assert counters["outside_challenge_windows"] == 1
    assert counters["not_on_foot"] == 1
    assert counters["excluded_home_proximate_endpoint"] == 1


def test_cluster_endpoint_records_rolls_up_reused_parking_anchor():
    module = load_module()
    records = [
        {"activity_id": 1, "activity_date": "2025-06-20", "window_label": "w", "endpoint_kind": "start", "lon": -116.2, "lat": 43.6},
        {"activity_id": 2, "activity_date": "2025-06-21", "window_label": "w", "endpoint_kind": "end", "lon": -116.2005, "lat": 43.6005},
        {"activity_id": 3, "activity_date": "2025-06-22", "window_label": "w", "endpoint_kind": "start", "lon": -116.5, "lat": 43.9},
    ]

    clusters = module.cluster_endpoint_records(records, cluster_radius_miles=0.08)

    assert len(clusters) == 2
    assert clusters[0]["evidence_activity_count"] == 2
    assert module.confidence_for_cluster(clusters[0]) == "strava_seen_prior_challenge_window"


def test_derive_parking_anchors_writes_private_geojson_shape(tmp_path):
    module = load_module()
    pull_dir = tmp_path / "pull"
    pull_dir.mkdir()
    (pull_dir / "activities_summary.json").write_text(
        json.dumps([activity(1, "2025-06-20", [43.61, -116.21], [43.61, -116.21])]),
        encoding="utf-8",
    )
    (pull_dir / "pull_summary.json").write_text(
        json.dumps(
            {
                "detail_selection": {
                    "challenge_windows": {
                        "2025_challenge_window": {"start": "2025-06-19", "end": "2025-07-19"}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"drive_model": {"origin_lat": 40, "origin_lon": -110}}), encoding="utf-8")

    geojson, summary = module.derive_parking_anchors(
        pull_dir / "activities_summary.json",
        pull_dir / "pull_summary.json",
        state,
        cluster_radius_miles=0.08,
        exclude_home_radius_miles=0.25,
        min_activity_miles=1.0,
    )

    assert geojson["features"][0]["properties"]["has_parking"] is True
    assert geojson["features"][0]["properties"]["privacy"] == "private_exact_coordinates"
    assert summary["anchor_count"] == 1
