import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "route_efficiency_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("route_efficiency_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_audit_marks_manual_hold_and_high_ratio_not_proven():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.7,
        },
        "packages": [
            {
                "package_number": 10,
                "block_name": "North pod",
                "components": [
                    {
                        "label": "10",
                        "trailhead": "Dry Creek",
                        "trail_names": ["Spring Creek"],
                        "official_miles": 9.0,
                        "on_foot_miles": 23.0,
                        "total_minutes": 400,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "10", "trailhead": "Dry Creek", "official_miles": 9.0, "on_foot_miles": 23.0}}
        ],
        "manual_holds": [{"label": "16A"}],
    }
    human_plan = {"summary": {"manual_design_area_count": 1, "planwide_on_foot_to_official_ratio": 1.7}}
    package16 = {
        "areas": [
            {
                "status": "accepted_split_probe_parking_manual",
                "current_placeholder": {"official_miles": 11.62, "on_foot_miles": 36.48},
                "current_best_split_probe": {
                    "official_miles": 11.62,
                    "on_foot_miles": 27.16,
                    "remaining_blocker": "parking manual",
                },
            }
        ]
    }

    audit = module.build_audit(map_data, field_packet, human_plan, package16)

    assert audit["achieved"] is False
    assert audit["verdict"] == "not_proven"
    statuses = {gate["gate"]: gate["status"] for gate in audit["gates"]}
    assert statuses["Full official coverage is represented"] == "passed"
    assert statuses["Runnable field packet covers all official work"] == "failed"
    assert statuses["No route exceeds 2.0x without a proven better-alternative comparison"] == "failed"


def test_build_audit_can_pass_when_efficiency_gates_are_met():
    module = load_module()
    map_data = {
        "summary": {
            "covered_segment_count": 251,
            "official_miles": 164.43,
            "planwide_on_foot_to_official_ratio": 1.45,
        },
        "packages": [
            {
                "package_number": 1,
                "block_name": "Good loop",
                "components": [
                    {
                        "label": "1",
                        "trailhead": "Trailhead",
                        "trail_names": ["Good Trail"],
                        "official_miles": 10.0,
                        "on_foot_miles": 14.0,
                        "total_minutes": 120,
                    }
                ],
            }
        ],
    }
    field_packet = {
        "summary": {},
        "routes": [
            {"outing": {"label": "1", "trailhead": "Trailhead", "official_miles": 10.0, "on_foot_miles": 14.0}}
        ],
        "manual_holds": [],
    }
    human_plan = {"summary": {"manual_design_area_count": 0, "planwide_on_foot_to_official_ratio": 1.45}}
    package16 = {"areas": []}

    audit = module.build_audit(map_data, field_packet, human_plan, package16)

    assert audit["achieved"] is True
    assert audit["verdict"] == "proven"


def test_render_md_includes_worst_components_and_next_work():
    module = load_module()
    audit = {
        "objective": "prove routes",
        "verdict": "not_proven",
        "achieved": False,
        "summary": {
            "all_component_totals": {"official_miles": 1, "on_foot_miles": 2, "ratio": 2.0},
            "runnable_field_packet_totals": {"official_miles": 1, "on_foot_miles": 2, "ratio": 2.0},
            "manual_hold_count": 1,
            "human_loop_plan_on_foot_miles": 2,
            "human_loop_plan_ratio": 2.0,
        },
        "gates": [{"gate": "A", "status": "failed", "evidence": "x"}],
        "package16": {},
        "worst_ratio_components": [
            {"label": "10", "trailhead": "Dry Creek", "official_miles": 1, "on_foot_miles": 3, "ratio": 3, "trails": ["Spring"]}
        ],
        "worst_overhead_components": [
            {
                "label": "10",
                "trailhead": "Dry Creek",
                "official_miles": 1,
                "on_foot_miles": 3,
                "overhead_miles": 2,
                "ratio": 3,
                "trails": ["Spring"],
            }
        ],
        "next_required_work": ["Challenge Dry Creek"],
    }

    rendered = module.render_md(audit)

    assert "Route Efficiency Audit" in rendered
    assert "Dry Creek" in rendered
    assert "Challenge Dry Creek" in rendered
