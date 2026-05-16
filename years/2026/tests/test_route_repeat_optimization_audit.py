import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "route_repeat_optimization_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("route_repeat_optimization_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def segment(seg_id, coords, direction="both"):
    return {
        "seg_id": seg_id,
        "seg_name": f"Segment {seg_id}",
        "trail_name": f"Trail {seg_id}",
        "official_miles": 0.5,
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


def build_audit(tmp_path, route, official_segments, route_proofs=None):
    module = load_module()
    return module.build_route_repeat_optimization_audit(
        {"routes": [route]},
        official_segments=official_segments,
        packet_dir=tmp_path / "packet",
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
