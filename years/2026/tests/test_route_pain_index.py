import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "route_pain_index.py"


def load_module():
    spec = importlib.util.spec_from_file_location("route_pain_index", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route(
    label,
    *,
    official=5.0,
    on_foot=8.0,
    p75=180,
    p90=220,
    parking=True,
    car_pass=False,
    water=False,
    warnings=0,
):
    return {
        "label": label,
        "trailhead": f"{label} trailhead",
        "official_miles": official,
        "on_foot_miles": on_foot,
        "door_to_door_minutes_p75": p75,
        "door_to_door_minutes_p90": p90,
        "parking": {"has_parking": parking},
        "logistics": {
            "has_car_pass": car_pass,
            "has_known_water": water,
        },
        "validation": {"passed": True},
        "wayfinding_cues": [
            {"field_warning": "repeat connector", "leg_miles": 1.0}
            for _ in range(warnings)
        ],
    }


def test_missing_verified_start_labels_parse_completion_audit_evidence():
    module = load_module()
    audit = {
        "checks": [
            {
                "passed": False,
                "evidence": (
                    "1A-1 Strava parking anchor 13: missing verified parked start; "
                    "5A West Hidden Springs Drive road-parking anchor: missing verified parked start"
                ),
            },
            {"passed": True, "evidence": "other check passed"},
        ]
    }

    assert module.missing_verified_start_labels(audit) == {"1A-1", "5A"}


def test_pain_index_prioritizes_current_access_sprint_over_stale_split_savings():
    module = load_module()
    field_payload = {
        "routes": [
            route("13", official=14.35, on_foot=25.12, p75=490, p90=549),
            route("10A", official=7.3, on_foot=13.62, p75=360, p90=404, car_pass=True, warnings=3),
            route("15A-1", official=6.97, on_foot=11.89, p75=229, p90=257, parking=False),
            route("15A-2", official=2.61, on_foot=4.35, p75=112, p90=140, parking=False),
        ]
    }
    multi_start_audit = {
        "outings": [
            {
                "label": "13",
                "alternatives": [
                    {
                        "alternative_id": "13-MS-06",
                        "status": "not_worth_it",
                        "on_foot_savings_miles": -2.6,
                        "elapsed_delta_minutes": 77,
                        "parking_blockers": [],
                    }
                ],
            },
            {
                "label": "10A",
                "alternatives": [
                    {
                        "alternative_id": "10A-MS-08",
                        "status": "needs_parking_check",
                        "on_foot_savings_miles": 3.38,
                        "elapsed_delta_minutes": -43,
                        "parking_blockers": ["parking/access requires manual verification"],
                    }
                ],
            },
            {
                "label": "15A",
                "alternatives": [
                    {
                        "alternative_id": "15A-MS-03",
                        "status": "promising",
                        "on_foot_savings_miles": 2.41,
                        "elapsed_delta_minutes": -38,
                        "parking_blockers": [],
                    }
                ],
            },
        ]
    }
    completion_audit = {
        "checks": [
            {
                "passed": False,
                "evidence": (
                    "15A-1 Dry Creek / Sweet Connie roadside parking: missing verified parked start; "
                    "15A-2 Bob's: missing verified parked start"
                ),
            }
        ]
    }

    report = module.build_pain_index(
        field_payload,
        multi_start_audit=multi_start_audit,
        completion_audit=completion_audit,
        generated_at="test",
    )

    assert report["summary"]["route_count"] == 4
    assert report["top_pain_routes"][0]["label"] == "13"
    assert report["top_actionable_optimizations"][0]["label"] == "10A"
    assert report["top_actionable_optimizations"][0]["optimization_status"] == "access_verification_sprint"
    assert report["top_actionable_optimizations"][0]["best_alternative"]["alternative_id"] == "10A-MS-08"
    assert report["route_rows_by_label"]["15A-1"]["optimization_status"] == "parked_start_certification"


def test_access_verification_blocks_paper_split_and_keeps_redesign_target():
    module = load_module()
    field_payload = {
        "routes": [
            route("13", official=14.35, on_foot=25.12, p75=490, p90=549),
            route("10A", official=7.3, on_foot=13.62, p75=360, p90=404, car_pass=True, warnings=3),
        ]
    }
    multi_start_audit = {
        "outings": [
            {
                "label": "10A",
                "alternatives": [
                    {
                        "alternative_id": "10A-MS-08",
                        "status": "needs_parking_check",
                        "on_foot_savings_miles": 3.38,
                        "elapsed_delta_minutes": -43,
                        "parking_blockers": ["parking/access requires manual verification"],
                    }
                ],
            }
        ]
    }
    access_verification = {
        "access_decisions": [
            {
                "alternative_id": "10A-MS-08",
                "decision": "not_certifiable",
                "field_certifiable": False,
                "replacement_ready": False,
                "next_action": "redesign_from_certifiable_parking",
                "evidence_file": "years/2026/checkpoints/10a-ms-08-access-verification-2026-05-10.md",
            }
        ]
    }

    report = module.build_pain_index(
        field_payload,
        multi_start_audit=multi_start_audit,
        access_verification=access_verification,
        generated_at="test",
    )

    ten_a = report["route_rows_by_label"]["10A"]
    assert ten_a["optimization_status"] == "certifiable_anchor_redesign"
    assert ten_a["best_alternative"]["access_decision"]["decision"] == "not_certifiable"
    assert report["summary"]["access_verification_sprint_count"] == 0
    assert report["summary"]["known_actionable_on_foot_savings_miles"] == 0.0
    assert report["summary"]["blocked_paper_on_foot_savings_miles"] == 3.38
    assert report["primary_recommendation"].startswith("Redesign 10A around a certifiable parked start")
