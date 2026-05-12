import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cluster_level_repricing_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cluster_level_repricing_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route(outing_id, label, segment_ids, actual_full_ids, on_foot, p75, lon=-116.0, lat=43.0):
    return {
        "route": {
            "outing_id": outing_id,
            "label": label,
            "candidate_ids": [label.lower()],
            "trailhead": f"{label} Trailhead",
            "segment_ids": segment_ids,
            "official_miles": len(segment_ids),
            "on_foot_miles": on_foot,
            "door_to_door_minutes_p75": p75,
            "door_to_door_minutes_p90": p75 + 10,
            "parking": {"lon": lon, "lat": lat},
        },
        "repeat_row": {
            "outing_id": outing_id,
            "label": f"{outing_id}: {label}",
            "actual_full_segment_ids": actual_full_ids,
            "declared_repeat_segment_ids": [],
        },
    }


def test_cluster_optimizer_removes_route_covered_by_source_latent_credit():
    module = load_module()
    a = route("a", "A", ["1"], ["1", "2"], 5.0, 50)
    b = route("b", "B", ["2"], ["2"], 3.0, 30)
    field_tool_data = {"routes": [a["route"], b["route"]]}
    latent_audit = {
        "reconciled_routes": [
            {
                "outing_id": "a",
                "label": "A",
                "segments": [
                    {
                        "seg_id": "2",
                        "status": "reconciled_owned_elsewhere",
                        "official_miles": 1.0,
                        "claimed_by_other_routes": [{"outing_id": "b", "label": "B"}],
                    }
                ],
            }
        ]
    }
    route_repeat_audit = {"routes": [a["repeat_row"], b["repeat_row"]]}
    official_segments = [
        {"seg_id": "1", "official_miles": 1.0},
        {"seg_id": "2", "official_miles": 1.0},
    ]

    audit = module.build_cluster_level_repricing_audit(
        field_tool_data,
        latent_audit,
        route_repeat_audit,
        official_segments,
        proximity_threshold_miles=0,
    )

    assert audit["status"] == "optimized_existing_loop_clusters"
    component = audit["components_with_savings"][0]
    assert component["current"]["route_count"] == 2
    assert component["optimized"]["route_count"] == 1
    assert component["selected_routes"][0]["outing_id"] == "a"
    assert component["skipped_routes"][0]["outing_id"] == "b"
    assert component["savings"]["on_foot_miles"] == 3.0


def test_optimizer_prefers_two_short_loops_over_one_long_covering_loop():
    module = load_module()
    a = route("a", "A", ["1", "2"], ["1", "2"], 10.0, 100)
    b = route("b", "B", ["1"], ["1"], 3.0, 30)
    c = route("c", "C", ["2"], ["2"], 4.0, 40)
    field_tool_data = {"routes": [a["route"], b["route"], c["route"]]}
    latent_audit = {
        "reconciled_routes": [
            {
                "outing_id": "a",
                "label": "A",
                "segments": [
                    {
                        "seg_id": "1",
                        "status": "reconciled_owned_elsewhere",
                        "official_miles": 1.0,
                        "claimed_by_other_routes": [{"outing_id": "b", "label": "B"}],
                    },
                    {
                        "seg_id": "2",
                        "status": "reconciled_owned_elsewhere",
                        "official_miles": 1.0,
                        "claimed_by_other_routes": [{"outing_id": "c", "label": "C"}],
                    },
                ],
            }
        ]
    }
    route_repeat_audit = {"routes": [a["repeat_row"], b["repeat_row"], c["repeat_row"]]}

    audit = module.build_cluster_level_repricing_audit(
        field_tool_data,
        latent_audit,
        route_repeat_audit,
        [{"seg_id": "1", "official_miles": 1.0}, {"seg_id": "2", "official_miles": 1.0}],
        proximity_threshold_miles=0,
    )

    component = audit["components"][0]
    assert component["optimization_status"] == "optimized_exact"
    assert {row["outing_id"] for row in component["selected_routes"]} == {"b", "c"}
    assert component["optimized"]["on_foot_miles"] == 7.0
