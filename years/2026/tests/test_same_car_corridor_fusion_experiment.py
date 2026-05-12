import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "same_car_corridor_fusion_experiment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("same_car_corridor_fusion_experiment", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route(label, outing_id, segment_ids, repeat_ids=None, owned_elsewhere=None, p75=30, p90=40, miles=1.0):
    return {
        "label": label,
        "outing_id": outing_id,
        "trailhead": "Test TH",
        "segment_ids": segment_ids,
        "official_miles": len(segment_ids) * 0.5,
        "on_foot_miles": miles,
        "door_to_door_minutes_p75": p75,
        "door_to_door_minutes_p90": p90,
        "candidate_ids": [label.lower()],
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "start_access",
                "signed_as": ["Shared Access"],
                "target": label,
                "route_leg_miles": 0.5,
                "confidence": "planner",
                "official_repeat_segment_ids": repeat_ids or [],
            },
            {
                "seq": 2,
                "cue_type": "follow_official_segment",
                "signed_as": [label],
                "target": "return",
                "route_leg_miles": miles - 0.5,
                "official_segment_ids": segment_ids,
            },
        ],
        "segment_ownership_reconciliation": {
            "declared_owned_elsewhere_segment_ids": owned_elsewhere or [],
        },
    }


def day(*routes):
    return {
        "field_day_id": "test-day",
        "date": "2026-07-04",
        "field_day_schedule_p75_minutes": 120,
        "field_day_schedule_p90_minutes": 150,
        "loops": [
            {
                "label": route["label"],
                "route_card_ref": {"outing_id": route["outing_id"], "label": route["label"]},
                "field_day_schedule_p75_minutes": route["door_to_door_minutes_p75"],
                "field_day_schedule_p90_minutes": route["door_to_door_minutes_p90"],
                "on_foot_miles": route["on_foot_miles"],
                "official_miles": route["official_miles"],
            }
            for route in routes
        ],
    }


def cluster_corridor(kind, routes):
    return {
        "kind": kind,
        "family_signature": f"{kind}|test|shared",
        "same_trailhead": True,
        "same_day_bundle_possible": True,
        "total_corridor_leg_miles": sum(row["leg_miles"] for row in routes),
        "routes": routes,
    }


def official_geojson(*segment_ids):
    return {
        "features": [
            {
                "properties": {
                    "segId": int(segment_id),
                    "segName": f"Segment {segment_id}",
                    "direction": "both",
                    "LengthFt": 1000,
                    "specInst": "",
                }
            }
            for segment_id in segment_ids
        ]
    }


def test_corridor_savings_lower_bound_keeps_longest_leg():
    module = load_module()

    result = module.corridor_savings_lower_bound(
        {"routes": [{"leg_miles": 1.84}, {"leg_miles": 2.5}]}
    )

    assert result["status"] == "lower_bound_duplicate_corridor_savings"
    assert result["miles"] == 1.84
    assert result["retained_corridor_leg_miles"] == 2.5


def test_promotion_probe_uses_source_physical_coverage_and_reprices_day(monkeypatch):
    module = load_module()
    owner = route("Owner", "owner", ["1"], p75=40, p90=50, miles=1.0)
    source = route("Source", "source", ["2"], repeat_ids=["1"], owned_elsewhere=["1"], p75=80, p90=90, miles=3.0)
    monkeypatch.setattr(
        module,
        "PROBES",
        [
            {
                "probe_id": "test-promotion",
                "title": "Test promotion",
                "route_labels": ["Owner", "Source"],
                "corridor_kind": "access",
                "strategy": "promote_source_claim_and_remove_owner",
                "source_label": "Source",
                "remove_labels": ["Owner"],
                "candidate_route_name": "Source claims Owner",
                "candidate_route_spec": ["Keep Source and promote Owner segment."],
            }
        ],
    )
    field_tool_data = {"routes": [owner, source], "field_day_layer": {"field_days": [day(owner, source)]}}
    cluster_audit = {
        "already_paid_access_corridors": [
            cluster_corridor(
                "access",
                [
                    {"route_key": "owner", "route": {"label": "owner: Owner"}, "leg_miles": 0.5},
                    {"route_key": "source", "route": {"label": "source: Source"}, "leg_miles": 0.7},
                ],
            )
        ]
    }
    repeat_audit = {
        "routes": [
            {"route_key": "owner", "actual_full_segment_ids": ["1"]},
            {"route_key": "source", "actual_full_segment_ids": ["1", "2"]},
        ]
    }

    report = module.build_report(field_tool_data, cluster_audit, repeat_audit, official_geojson("1", "2"))

    probe = report["probes"][0]
    candidate = probe["candidate_fused_route"]
    assert candidate["status"] == "promotion_candidate_requires_recertification"
    assert candidate["official_segment_coverage"]["status"] == "coverage_preserved_if_source_claim_promoted"
    assert candidate["field_day_after_removal"]["remaining_loop_labels"] == ["Source"]
    assert candidate["saved_on_foot_miles"] == 1.0


def test_paper_fusion_gets_scaled_time_and_non_promotion_gate(monkeypatch):
    module = load_module()
    first = route("First", "first", ["1"], p75=60, p90=70, miles=2.0)
    second = route("Second", "second", ["2"], p75=90, p90=100, miles=4.0)
    monkeypatch.setattr(
        module,
        "PROBES",
        [
            {
                "probe_id": "test-paper",
                "title": "Test paper",
                "route_labels": ["First", "Second"],
                "corridor_kind": "return",
                "strategy": "drop_intermediate_return_corridor",
                "candidate_route_name": "Paper fusion",
                "candidate_route_spec": ["Drop one return corridor."],
            }
        ],
    )
    field_tool_data = {"routes": [first, second], "field_day_layer": {"field_days": [day(first, second)]}}
    cluster_audit = {
        "already_paid_access_corridors": [
            cluster_corridor(
                "return",
                [
                    {"route_key": "first", "route": {"label": "first: First"}, "leg_miles": 1.0},
                    {"route_key": "second", "route": {"label": "second: Second"}, "leg_miles": 2.0},
                ],
            )
        ]
    }

    report = module.build_report(field_tool_data, cluster_audit, {"routes": []}, official_geojson("1", "2"))

    candidate = report["probes"][0]["candidate_fused_route"]
    assert candidate["status"] == "paper_only_needs_continuous_gpx_timing_and_coverage"
    assert candidate["saved_on_foot_miles"] == 1.0
    assert candidate["pricing_status"] == "corridor_scaled_estimate_needs_dem_and_gpx"
    assert candidate["promotion_status"] == "not_promotable_until_continuous_gpx_p75_cues_coverage_and_recertification"
