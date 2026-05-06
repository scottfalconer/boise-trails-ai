from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_responsible_relaxed_certificate as certificate  # noqa: E402


def official_geojson(ids):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"segId": seg_id}, "geometry": None}
            for seg_id in ids
        ],
    }


def day(segment_ids, *, on_foot=5.0, p90=120, bound=292):
    return {
        "draft_day_number": 1,
        "day_type": "weekday",
        "p75_minutes": 100,
        "p90_minutes": p90,
        "p90_bound_minutes": bound,
        "between_drive_minutes": 0,
        "on_foot_miles": on_foot,
        "segment_summary": {"segment_ids": segment_ids},
        "loops": [
            {
                "loop_id": "loop-1",
                "trailhead": "Test Trailhead",
                "validation_passed": True,
                "manual_design_hold": False,
                "parking_confidence": "source_verified",
            }
        ],
    }


def plan(days):
    covered = sorted({seg_id for item in days for seg_id in item["segment_summary"]["segment_ids"]})
    return {
        "coverage": {"official_segment_count": len(covered), "covered_segment_count": len(covered)},
        "field_days": days,
    }


def calendar(days):
    covered = sorted({seg_id for item in days for seg_id in item["segment_summary"]["segment_ids"]})
    return {
        "audit": {
            "assigned_day_count": len(days),
            "covered_segment_count": len(covered),
            "missing_segment_count": 0,
            "day_type_violation_count": 0,
            "lower_hulls_even_day_violation_count": 0,
            "p90_violation_count": 0,
            "passed": True,
        },
        "assignments": [{"field_day": item} for item in days],
    }


def gpx():
    return {
        "summary": {
            "day_gpx_count": 1,
            "loop_validation_passed": True,
            "day_track_validation_passed": True,
            "failed_day_count": 0,
            "max_gap_miles": 0.05,
            "max_endpoint_gap_miles": 0.35,
        },
        "days": [
            {
                "day_track_validation": {"max_trackpoint_gap_miles": 0.01},
                "loops": [{"validation": {"endpoint_gap_miles": 0.0}}],
            }
        ],
    }


def pressure():
    return {
        "field_day_candidate_count": 12,
        "p75_min_full_cover": {
            "success": True,
            "field_day_count": 1,
            "total_p75_minutes": 100,
            "total_on_foot_miles_day_sum": 5.0,
        },
    }


def lower_bound():
    return {
        "method": "test lower bound",
        "summary": {
            "official_miles": 2.0,
            "connector_graph_lower_bound_miles": 4.0,
        },
    }


def build(days, official_ids=(1, 2)):
    return certificate.build_report(
        profile=certificate.default_profile(),
        official_geojson=official_geojson(official_ids),
        plan=plan(days),
        calendar=calendar(days),
        gpx=gpx(),
        pressure=pressure(),
        lower_bound=lower_bound(),
        private_state={
            "trailheads": [
                {
                    "name": "Test Trailhead",
                    "has_parking": True,
                    "facility_status": "Open",
                    "parking_confidence": "osm_amenity_parking_near_official_start",
                }
            ]
        },
        trailhead_candidates={},
        parking_access={},
        strava_parking={},
        paths={},
    )


def test_certificate_passes_when_all_required_segments_fit_profile():
    report = build([day([1, 2])])

    assert report["certificate_status"] == "passed"
    assert report["segment_set"]["missing_segment_count"] == 0
    assert report["field_days"]["max_on_foot_miles"] == 5.0


def test_missing_required_segment_fails_certificate():
    report = build([day([1])], official_ids=(1, 2))

    assert report["certificate_status"] == "failed"
    assert report["segment_set"]["missing_segment_ids"] == [2]
    gate = next(row for row in report["gates"] if row["gate"] == "all_official_segments_required")
    assert gate["passed"] is False


def test_day_over_18_miles_fails_certificate():
    report = build([day([1, 2], on_foot=18.01)])

    assert report["certificate_status"] == "failed"
    gate = next(row for row in report["gates"] if row["gate"] == "on_foot_18_mile_daily_cap")
    assert gate["passed"] is False


def test_unverified_parked_start_fails_certificate():
    report = certificate.build_report(
        profile=certificate.default_profile(),
        official_geojson=official_geojson((1, 2)),
        plan=plan([day([1, 2])]),
        calendar=calendar([day([1, 2])]),
        gpx=gpx(),
        pressure=pressure(),
        lower_bound=lower_bound(),
        private_state={},
        trailhead_candidates={},
        parking_access={},
        strava_parking={},
        paths={},
    )

    assert report["certificate_status"] == "failed"
    gate = next(row for row in report["gates"] if row["gate"] == "legal_parked_starts")
    assert gate["passed"] is False


def test_parking_checkpoint_can_verify_manual_roadside_anchor():
    anchors = certificate.parking_anchor_index(
        private_state={},
        trailhead_candidates={},
        parking_access={
            "anchors": [
                {
                    "name": "Manual Roadside",
                    "status": "source_verified_for_planning",
                }
            ]
        },
        strava_parking={},
    )

    verification = certificate.parked_start_verification(
        [{"loops": [{"trailhead": "Manual Roadside"}]}],
        anchors,
    )

    assert verification["all_parked_starts_verified"] is True


def test_loop_endpoint_gap_fails_same_car_gate():
    bad_gpx = gpx()
    bad_gpx["days"][0]["loops"][0]["validation"]["endpoint_gap_miles"] = 0.1
    report = certificate.build_report(
        profile=certificate.default_profile(),
        official_geojson=official_geojson((1, 2)),
        plan=plan([day([1, 2])]),
        calendar=calendar([day([1, 2])]),
        gpx=bad_gpx,
        pressure=pressure(),
        lower_bound=lower_bound(),
        private_state={
            "trailheads": [
                {
                    "name": "Test Trailhead",
                    "has_parking": True,
                    "facility_status": "Open",
                    "parking_confidence": "osm_amenity_parking_near_official_start",
                }
            ]
        },
        trailhead_candidates={},
        parking_access={},
        strava_parking={},
        paths={},
    )

    assert report["certificate_status"] == "failed"
    gate = next(row for row in report["gates"] if row["gate"] == "same_car_loop_endpoints")
    assert gate["passed"] is False
