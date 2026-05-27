import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "route_repeat_optimization_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("route_repeat_optimization_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def segment(seg_id, coords, direction="both", official_miles=0.5):
    return {
        "seg_id": seg_id,
        "seg_name": f"Segment {seg_id}",
        "trail_name": f"Trail {seg_id}",
        "official_miles": official_miles,
        "direction": direction,
        "coordinates": coords,
        "start": coords[0],
        "end": coords[-1],
    }


def write_gpx(path, coords):
    path.parent.mkdir(parents=True, exist_ok=True)
    points = "\n".join(f'<trkpt lat="{lat}" lon="{lon}"></trkpt>' for lon, lat in coords)
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
  <trk><trkseg>
{points}
  </trkseg></trk>
</gpx>
""",
        encoding="utf-8",
    )

def write_connector_geojson(path, features):
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def connector_feature(name, coords):
    return {
        "type": "Feature",
        "properties": {"TrailName": name},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def path_miles(module, coords):
    return sum(module.haversine_miles(left, right) for left, right in zip(coords, coords[1:]))


def build_audit(tmp_path, route, official_segments, route_proofs=None, progress=None, connector_features=None):
    return build_audit_for_routes(
        tmp_path,
        [route],
        official_segments,
        route_proofs=route_proofs,
        progress=progress,
        connector_features=connector_features,
    )


def build_audit_for_routes(tmp_path, routes, official_segments, route_proofs=None, progress=None, connector_features=None):
    module = load_module()
    connector_path = None
    if connector_features is not None:
        connector_path = tmp_path / "connectors.geojson"
        write_connector_geojson(connector_path, connector_features)
    return module.build_route_repeat_optimization_audit(
        {"routes": routes, "progress": progress or {}},
        official_segments=official_segments,
        packet_dir=tmp_path / "packet",
        connector_graph_path=connector_path,
        threshold_miles=0.015,
        min_fraction=0.8,
        route_proofs=route_proofs,
    )


def test_hidden_self_repeat_fails_when_non_credit_leg_reuses_claimed_segment(tmp_path):
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-a.gpx",
        [(-116.0, 43.0), (-115.99, 43.0), (-116.0, 43.0)],
    )
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "trailhead": "Trailhead A",
        "segment_ids": [101],
        "audit_gpx_href": "gpx/audit/route-a.gpx",
        "official_miles": 0.5,
        "on_foot_miles": 1.0,
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "route_miles": 0.0, "route_leg_miles": 0.51},
            {"seq": 2, "cue_type": "exit_access", "route_miles": 0.51, "route_leg_miles": 0.51},
        ],
    }

    audit = build_audit(
        tmp_path,
        route,
        [segment(101, [(-116.0, 43.0), (-115.99, 43.0)])],
    )

    assert audit["status"] == "failed"
    assert audit["hard_failures"]["hidden_self_repeat_segment_ids"] == ["101"]
    assert audit["failed_routes"][0]["hidden_self_repeat_ids"] == ["101"]
    assert audit["advisory_closure"]["status"] == "blocked_by_hard_failures"


def test_latent_credit_without_ownership_decision_fails(tmp_path):
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-a.gpx",
        [
            (-116.0, 43.0),
            (-115.99, 43.0),
            (-116.0, 43.02),
            (-115.99, 43.02),
        ],
    )
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "trailhead": "Trailhead A",
        "segment_ids": [101],
        "audit_gpx_href": "gpx/audit/route-a.gpx",
        "official_miles": 0.5,
        "on_foot_miles": 1.4,
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "route_miles": 0.0, "route_leg_miles": 0.51},
        ],
    }

    audit = build_audit(
        tmp_path,
        route,
        [
            segment(101, [(-116.0, 43.0), (-115.99, 43.0)]),
            segment(102, [(-116.0, 43.02), (-115.99, 43.02)]),
        ],
    )

    assert audit["status"] == "failed"
    assert audit["hard_failures"]["latent_credit_segment_ids"] == ["102"]
    assert audit["failed_routes"][0]["latent_credit_ids"] == ["102"]


def test_declared_priced_repeat_passes(tmp_path):
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-a.gpx",
        [(-116.0, 43.0), (-115.99, 43.0), (-116.0, 43.0)],
    )
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "trailhead": "Trailhead A",
        "segment_ids": [101],
        "audit_gpx_href": "gpx/audit/route-a.gpx",
        "official_miles": 0.5,
        "on_foot_miles": 1.0,
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "route_miles": 0.0, "route_leg_miles": 0.51},
            {
                "seq": 2,
                "cue_type": "exit_access",
                "route_miles": 0.51,
                "route_leg_miles": 0.51,
                "official_repeat_segment_ids": [101],
                "official_repeat_miles": 0.51,
                "note": "Includes 0.51 mi repeat official; no new credit.",
            },
        ],
    }

    audit = build_audit(
        tmp_path,
        route,
        [segment(101, [(-116.0, 43.0), (-115.99, 43.0)])],
    )

    assert audit["status"] == "passed"
    assert audit["summary"]["failed_route_count"] == 0
    assert audit["hard_failures"]["unpriced_repeat_segment_ids"] == []


def test_optimization_warnings_are_closed_as_non_blocking_when_repeat_accounting_passes(tmp_path):
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-a.gpx",
        [(-116.0, 43.0), (-115.99, 43.0), (-116.0, 43.0)],
    )
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "trailhead": "Trailhead A",
        "segment_ids": [101],
        "audit_gpx_href": "gpx/audit/route-a.gpx",
        "official_miles": 0.5,
        "on_foot_miles": 2.0,
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "route_miles": 0.0, "route_leg_miles": 0.51},
            {
                "seq": 2,
                "cue_type": "exit_access",
                "route_miles": 0.51,
                "route_leg_miles": 0.51,
                "official_repeat_segment_ids": [101],
                "official_repeat_miles": 0.51,
                "note": "Includes 0.51 mi repeat official; no new credit.",
            },
        ],
    }

    audit = build_audit(
        tmp_path,
        route,
        [segment(101, [(-116.0, 43.0), (-115.99, 43.0)])],
    )

    assert audit["status"] == "passed"
    assert audit["summary"]["optimization_warning_count"] == 1
    assert audit["summary"]["total_optimization_warning_count"] == 1
    assert audit["summary"]["closed_optimization_warning_count"] == 0
    assert audit["advisory_closure"]["status"] == "closed_non_blocking_optimization_backlog"
    assert audit["advisory_closure"]["warning_counts"] == {"high_on_foot_to_official_ratio": 1}


def test_proofed_optimization_warnings_are_closed_by_route_disproof(tmp_path):
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-a.gpx",
        [(-116.0, 43.0), (-115.99, 43.0), (-116.0, 43.0)],
    )
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "trailhead": "Trailhead A",
        "candidate_ids": ["accepted-grinder"],
        "segment_ids": [101],
        "audit_gpx_href": "gpx/audit/route-a.gpx",
        "official_miles": 0.5,
        "on_foot_miles": 2.0,
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "route_miles": 0.0, "route_leg_miles": 0.51},
            {
                "seq": 2,
                "cue_type": "exit_access",
                "route_miles": 0.51,
                "route_leg_miles": 0.51,
                "official_repeat_segment_ids": [101],
                "official_repeat_miles": 0.51,
                "note": "Includes 0.51 mi repeat official; no new credit.",
            },
        ],
    }
    route_proofs = [
        {
            "proofs": [
                {
                    "candidate_ids": ["accepted-grinder"],
                    "candidate_id": "accepted-grinder",
                    "area": "Accepted grinder",
                    "status": "accepted_current",
                    "checks": {
                        "gpx_continuity_passed": True,
                        "current_route_has_p75_time": True,
                        "current_route_has_dem_effort": True,
                        "no_better_exact_generated_candidate": True,
                        "no_dominant_boundary_recombination": True,
                        "no_dominant_global_optimizer_replacement": True,
                    },
                }
            ]
        }
    ]

    audit = build_audit(
        tmp_path,
        route,
        [segment(101, [(-116.0, 43.0), (-115.99, 43.0)])],
        route_proofs=route_proofs,
    )

    assert audit["status"] == "passed"
    assert audit["summary"]["optimization_warning_count"] == 0
    assert audit["summary"]["total_optimization_warning_count"] == 1
    assert audit["summary"]["closed_optimization_warning_count"] == 1
    assert audit["advisory_closure"]["status"] == "closed_by_route_disproof"
    assert audit["optimization_warnings"] == []
    assert audit["closed_optimization_warnings"][0]["proof_status"] == "closed_by_route_disproof"


def test_unpriced_declared_repeat_fails(tmp_path):
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-a.gpx",
        [(-116.0, 43.0), (-115.99, 43.0), (-116.0, 43.0)],
    )
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "trailhead": "Trailhead A",
        "segment_ids": [101],
        "audit_gpx_href": "gpx/audit/route-a.gpx",
        "official_miles": 0.5,
        "on_foot_miles": 1.0,
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "route_miles": 0.0, "route_leg_miles": 0.51},
            {
                "seq": 2,
                "cue_type": "exit_access",
                "route_miles": 0.51,
                "route_leg_miles": 0.51,
                "official_repeat_segment_ids": [101],
            },
        ],
    }

    audit = build_audit(
        tmp_path,
        route,
        [segment(101, [(-116.0, 43.0), (-115.99, 43.0)])],
    )

    assert audit["status"] == "failed"
    assert audit["hard_failures"]["unpriced_repeat_segment_ids"] == ["101"]
    assert audit["failed_routes"][0]["unpriced_repeat_rows"][0]["repeat_miles_missing_or_zero"] is True

def test_fd12a_shaped_post_credit_repeat_fails_when_shorter_connector_exists(tmp_path):
    module = load_module()
    packet_dir = tmp_path / "packet"
    path_out = [
        (-116.000, 43.000),
        (-116.000, 43.006),
        (-115.994, 43.006),
        (-115.994, 43.012),
        (-115.988, 43.012),
        (-115.988, 43.000),
    ]
    repeat_back = list(reversed(path_out[:-1]))
    route_coords = path_out + repeat_back
    write_gpx(packet_dir / "gpx" / "audit" / "fd12a-shaped.gpx", route_coords)
    earned_miles = path_miles(module, path_out)
    repeat_miles = path_miles(module, [path_out[-1], *repeat_back])
    segment_ids = [1504, 1505, 1506, 1507, 1755]
    official_segments = [
        segment(segment_id, [left, right])
        for segment_id, left, right in zip(segment_ids, path_out, path_out[1:])
    ]
    route = {
        "outing_id": "112-1",
        "label": "FD12A",
        "trailhead": "West Climb",
        "segment_ids": segment_ids,
        "audit_gpx_href": "gpx/audit/fd12a-shaped.gpx",
        "official_miles": earned_miles,
        "on_foot_miles": earned_miles + repeat_miles,
        "wayfinding_cues": [
            {
                "seq": 7,
                "cue_type": "junction_turn",
                "route_miles": 0.0,
                "route_leg_miles": earned_miles,
                "official_segment_ids": segment_ids,
                "note": "This earns: Buena Vista Trail segments 1-5.",
            },
            {
                "seq": 8,
                "cue_type": "exit_access",
                "route_miles": earned_miles,
                "route_leg_miles": repeat_miles,
                "official_repeat_segment_ids": segment_ids,
                "official_repeat_miles": repeat_miles,
                "note": "Return leg includes repeat official; no new credit.",
            },
        ],
    }
    connector_path = tmp_path / "connectors.geojson"
    write_connector_geojson(
        connector_path,
        [connector_feature("Full Sail continuation", [path_out[-1], path_out[0]])],
    )

    audit = module.build_route_repeat_optimization_audit(
        {"routes": [route], "progress": {"completed_segment_ids_at_export": []}},
        official_segments=official_segments,
        packet_dir=packet_dir,
        connector_graph_path=connector_path,
        threshold_miles=0.015,
        min_fraction=0.8,
    )

    assert audit["status"] == "failed"
    assert audit["summary"]["avoidable_post_credit_repeat_instance_count"] == 1
    assert audit["hard_failures"]["avoidable_post_credit_repeat_segment_ids"] == [str(i) for i in segment_ids]
    instance = audit["failed_routes"][0]["avoidable_post_credit_repeat_instances"][0]
    assert instance["seq"] == 8
    assert instance["repeated_segment_ids"] == [str(i) for i in segment_ids]
    assert instance["alternate_connector_names"] == ["Full Sail continuation"]


def test_declared_repeat_without_alternate_path_emits_advisory_not_hard_failure(tmp_path):
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-a.gpx",
        [(-116.0, 43.0), (-115.99, 43.0), (-116.0, 43.0)],
    )
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "trailhead": "Trailhead A",
        "segment_ids": [101],
        "audit_gpx_href": "gpx/audit/route-a.gpx",
        "official_miles": 0.5,
        "on_foot_miles": 1.0,
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "route_miles": 0.0, "route_leg_miles": 0.51, "official_segment_ids": [101]},
            {
                "seq": 2,
                "cue_type": "exit_access",
                "route_miles": 0.51,
                "route_leg_miles": 0.51,
                "official_repeat_segment_ids": [101],
                "official_repeat_miles": 0.51,
                "note": "Includes 0.51 mi repeat official; no new credit.",
            },
        ],
    }

    audit = build_audit(
        tmp_path,
        route,
        [segment(101, [(-116.0, 43.0), (-115.99, 43.0)])],
        connector_features=[],
    )

    assert audit["status"] == "passed"
    assert audit["summary"]["avoidable_post_credit_repeat_instance_count"] == 0
    assert audit["summary"]["post_credit_repeat_advisory_count"] == 1
    assert audit["routes"][0]["post_credit_repeat_advisories"][0]["reason"] == "repeat_exit_no_alternate_graph_path_proven"


def test_completed_segments_at_export_count_as_prior_credit_for_first_cue_repeat(tmp_path):
    packet_dir = tmp_path / "packet"
    repeated_route = [(-116.0, 43.0), (-115.995, 43.004), (-115.99, 43.0)]
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-a.gpx",
        repeated_route,
    )
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "trailhead": "Trailhead A",
        "segment_ids": [],
        "audit_gpx_href": "gpx/audit/route-a.gpx",
        "official_miles": 0.0,
        "on_foot_miles": 0.5,
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "start_access",
                "route_miles": 0.0,
                "route_leg_miles": 1.2,
                "official_repeat_segment_ids": [101],
                "official_repeat_miles": 0.51,
                "note": "Includes repeat official; no new credit.",
            }
        ],
    }

    audit = build_audit(
        tmp_path,
        route,
        [segment(101, repeated_route)],
        progress={"completed_segment_ids_at_export": [101]},
        connector_features=[
            connector_feature("Public shortcut", [[-116.0, 43.0], [-115.99, 43.0]])
        ],
    )

    assert audit["status"] == "failed"
    assert audit["summary"]["avoidable_post_credit_repeat_instance_count"] == 1
    assert audit["failed_routes"][0]["avoidable_post_credit_repeat_instances"][0]["already_credited_source"] == "completed_at_export"


def test_avoidable_repeat_savings_below_threshold_is_not_hard_failure(tmp_path):
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-a.gpx",
        [(-116.0, 43.0), (-115.99, 43.0), (-116.0, 43.0)],
    )
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "trailhead": "Trailhead A",
        "segment_ids": [101],
        "audit_gpx_href": "gpx/audit/route-a.gpx",
        "official_miles": 0.5,
        "on_foot_miles": 1.0,
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "route_miles": 0.0, "route_leg_miles": 0.51, "official_segment_ids": [101]},
            {
                "seq": 2,
                "cue_type": "exit_access",
                "route_miles": 0.51,
                "route_leg_miles": 0.51,
                "official_repeat_segment_ids": [101],
                "official_repeat_miles": 0.51,
                "note": "Includes 0.51 mi repeat official; no new credit.",
            },
        ],
    }

    audit = build_audit(
        tmp_path,
        route,
        [segment(101, [(-116.0, 43.0), (-115.99, 43.0)])],
        connector_features=[
            connector_feature("Tiny savings connector", [[-116.0, 43.0], [-115.9905, 43.0], [-115.99, 43.0]])
        ],
    )

    assert audit["status"] == "passed"
    assert audit["summary"]["avoidable_post_credit_repeat_instance_count"] == 0


def test_graph_scaled_alternate_that_is_physically_longer_is_not_hard_failure(tmp_path):
    start = (-116.0, 43.0)
    end = (-115.99, 43.0)
    long_detour = (-115.995, 43.02)
    packet_dir = tmp_path / "packet"
    write_gpx(packet_dir / "gpx" / "audit" / "route-a.gpx", [start, end, start])
    module = load_module()
    leg_miles = path_miles(module, [start, end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "trailhead": "Trailhead A",
        "segment_ids": [101],
        "audit_gpx_href": "gpx/audit/route-a.gpx",
        "official_miles": leg_miles,
        "on_foot_miles": leg_miles * 2,
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "route_miles": 0.0,
                "route_leg_miles": leg_miles,
                "official_segment_ids": [101],
            },
            {
                "seq": 2,
                "cue_type": "exit_access",
                "route_miles": leg_miles,
                "route_leg_miles": leg_miles,
                "official_repeat_segment_ids": [101],
                "official_repeat_miles": leg_miles,
                "note": "Includes repeat official; no new credit.",
            },
        ],
    }

    audit = build_audit(
        tmp_path,
        route,
        [
            segment(101, [start, end], official_miles=leg_miles),
            segment(202, [end, long_detour, start], official_miles=0.01),
        ],
    )

    assert audit["status"] == "passed"
    assert audit["summary"]["avoidable_post_credit_repeat_instance_count"] == 0


def test_cross_route_tail_opportunity_warns_for_small_adjacent_split_segment(tmp_path):
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-a.gpx",
        [(-116.0, 43.0), (-115.99, 43.0)],
    )
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-b.gpx",
        [(-116.0, 43.0), (-115.99, 43.0), (-115.98, 43.0)],
    )
    route_a = {
        "outing_id": "route-a",
        "label": "Route A",
        "trailhead": "Trailhead A",
        "segment_ids": [201],
        "audit_gpx_href": "gpx/audit/route-a.gpx",
        "official_miles": 0.2,
        "on_foot_miles": 0.2,
        "wayfinding_cues": [
            {"seq": 1, "cue_type": "follow_official_segment", "route_miles": 0.0, "route_leg_miles": 0.2},
        ],
    }
    route_b = {
        "outing_id": "route-b",
        "label": "Route B",
        "trailhead": "Trailhead B",
        "segment_ids": [202],
        "audit_gpx_href": "gpx/audit/route-b.gpx",
        "official_miles": 0.2,
        "on_foot_miles": 0.4,
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "connector_named_trail",
                "route_miles": 0.0,
                "route_leg_miles": 0.2,
                "official_repeat_segment_ids": [201],
                "official_repeat_miles": 0.2,
                "note": "Includes 0.2 mi repeat official; no new credit.",
            },
            {
                "seq": 2,
                "cue_type": "junction_turn",
                "route_miles": 0.2,
                "route_leg_miles": 0.2,
                "official_segment_ids": [202],
            },
        ],
        "segment_ownership_reconciliation": {
            "declared_owned_elsewhere_segment_ids": ["201"],
            "segments_owned_elsewhere": [
                {
                    "seg_id": "201",
                    "seg_name": "Segment 201",
                    "official_miles": 0.2,
                    "owned_by_routes": [{"outing_id": "route-a", "label": "Route A"}],
                }
            ],
        },
    }

    audit = build_audit_for_routes(
        tmp_path,
        [route_a, route_b],
        [
            segment(201, [(-116.0, 43.0), (-115.99, 43.0)], official_miles=0.2),
            segment(202, [(-115.99, 43.0), (-115.98, 43.0)], official_miles=0.2),
        ],
    )

    assert audit["status"] == "passed"
    assert audit["summary"]["cross_route_tail_opportunity_count"] == 1
    opportunity = audit["cross_route_tail_opportunities"][0]
    assert opportunity["receiver_outing_id"] == "route-b"
    assert opportunity["repeated_owned_segment"]["seg_id"] == "201"
    assert opportunity["adjacent_candidate_segment"]["seg_id"] == "202"
    assert opportunity["cue_context"]["nearest_cue_pair"]["repeat_cue_seq"] == 1
    assert opportunity["cue_context"]["nearest_cue_pair"]["adjacent_cue_seq"] == 2


def test_cross_route_tail_opportunity_ignores_large_adjacent_segments(tmp_path):
    packet_dir = tmp_path / "packet"
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-a.gpx",
        [(-116.0, 43.0), (-115.99, 43.0)],
    )
    write_gpx(
        packet_dir / "gpx" / "audit" / "route-b.gpx",
        [(-116.0, 43.0), (-115.99, 43.0), (-115.98, 43.0)],
    )
    route_a = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [301],
        "audit_gpx_href": "gpx/audit/route-a.gpx",
        "official_miles": 0.2,
        "on_foot_miles": 0.2,
    }
    route_b = {
        "outing_id": "route-b",
        "label": "Route B",
        "segment_ids": [302],
        "audit_gpx_href": "gpx/audit/route-b.gpx",
        "official_miles": 0.6,
        "on_foot_miles": 0.8,
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "connector_named_trail",
                "official_repeat_segment_ids": [301],
                "official_repeat_miles": 0.2,
                "note": "Includes 0.2 mi repeat official; no new credit.",
            }
        ],
        "segment_ownership_reconciliation": {
            "declared_owned_elsewhere_segment_ids": ["301"],
            "segments_owned_elsewhere": [
                {
                    "seg_id": "301",
                    "official_miles": 0.2,
                    "owned_by_routes": [{"outing_id": "route-a", "label": "Route A"}],
                }
            ],
        },
    }

    audit = build_audit_for_routes(
        tmp_path,
        [route_a, route_b],
        [
            segment(301, [(-116.0, 43.0), (-115.99, 43.0)], official_miles=0.2),
            segment(302, [(-115.99, 43.0), (-115.98, 43.0)], official_miles=0.6),
        ],
    )

    assert audit["summary"]["cross_route_tail_opportunity_count"] == 0
