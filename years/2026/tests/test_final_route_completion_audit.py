import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "final_route_completion_audit.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("final_route_completion_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_audit_marks_fragmented_candidate_pass_not_complete():
    module = load_audit_module()
    block_review = {
        "summary": {
            "official_segment_assignment_passed": True,
            "block_count": 19,
            "draft_block_count": 16,
        }
    }
    route_pass = {
        "summary": {
            "covered_segment_count": 251,
            "selected_route_count": 47,
            "routes_under_1_official_mile": 16,
            "routes_under_2_official_miles": 22,
            "planwide_on_foot_to_official_ratio": 1.98,
        }
    }
    route_map = {"map_validation": {"rendered_passed": True, "source_gap_warning_count": 6}}

    audit = module.build_audit(block_review, route_pass, route_map)

    assert audit["achieved"] is False
    assert audit["verdict"] == "not_complete"
    statuses = {item["requirement"]: item["status"] for item in audit["checklist"]}
    assert statuses["Full official 2026 on-foot coverage"] == "passed"
    assert statuses["Normal-human route quality with minimal car-hop fragments"] == "failed"


def test_build_audit_uses_package_and_assembled_diagnostics():
    module = load_audit_module()
    block_review = {
        "summary": {
            "official_segment_assignment_passed": True,
            "block_count": 19,
            "draft_block_count": 10,
        }
    }
    route_pass = {
        "summary": {
            "covered_segment_count": 251,
            "selected_route_count": 47,
            "routes_under_1_official_mile": 16,
            "routes_under_2_official_miles": 22,
            "total_on_foot_miles": 326.08,
            "planwide_on_foot_to_official_ratio": 1.98,
        }
    }
    route_map = {"map_validation": {"rendered_passed": True, "source_gap_warning_count": 6}}
    package_pass = {
        "summary": {
            "package_count": 18,
            "component_route_count": 47,
            "packages_with_multiple_trailheads": 10,
        }
    }
    package_map = {"map_validation": {"rendered_passed": True}}
    assembled_pass = {"summary": {"total_on_foot_miles": 388.35, "assembled_route_count": 19}}

    audit = module.build_audit(
        block_review,
        route_pass,
        route_map,
        package_pass=package_pass,
        package_map=package_map,
        assembled_pass=assembled_pass,
    )

    statuses = {item["requirement"]: item["status"] for item in audit["checklist"]}
    assert statuses["Actual real loop/block route structure"] == "failed"
    assert statuses["Hybrid selection beats naive one-route-per-block assembly"] == "passed"
    assert statuses["Single user-facing loop plan artifact"] == "failed"
    assert "human-loop-plan-v1.md" in audit["next_required_work"][0]


def test_build_audit_accepts_human_loop_plan_as_review_artifact():
    module = load_audit_module()
    block_review = {
        "summary": {
            "official_segment_assignment_passed": True,
            "block_count": 19,
            "draft_block_count": 10,
        }
    }
    route_pass = {
        "summary": {
            "covered_segment_count": 251,
            "selected_route_count": 29,
            "routes_under_1_official_mile": 4,
            "routes_under_2_official_miles": 8,
            "total_on_foot_miles": 308.55,
            "planwide_on_foot_to_official_ratio": 1.88,
        }
    }
    route_map = {"map_validation": {"rendered_passed": True, "source_gap_warning_count": 10}}
    package_pass = {
        "summary": {
            "package_count": 18,
            "component_route_count": 29,
            "packages_with_multiple_trailheads": 8,
        }
    }
    package_map = {"map_validation": {"rendered_passed": True}}
    human_plan = {
        "summary": {
            "covered_segment_count": 251,
            "unresolved_blocker_count": 0,
            "all_route_components_graph_validated": True,
            "map_rendered_passed": True,
            "status_counts": {
                "primary_loop_block": 3,
                "accepted_split_block": 6,
                "necessary_grinder": 9,
            },
        }
    }

    audit = module.build_audit(
        block_review,
        route_pass,
        route_map,
        package_pass=package_pass,
        package_map=package_map,
        human_plan=human_plan,
    )

    statuses = {item["requirement"]: item["status"] for item in audit["checklist"]}
    assert statuses["Single user-facing loop plan artifact"] == "passed"
    assert statuses["Ready to call final"] == "failed"


def test_render_markdown_includes_next_required_work():
    module = load_audit_module()
    audit = {
        "objective": "example",
        "achieved": False,
        "verdict": "not_complete",
        "checklist": [
            {"requirement": "A", "status": "failed", "evidence": ["x", "y"]}
        ],
        "next_required_work": ["Do custom GPX"],
    }

    rendered = module.render_markdown(audit)

    assert "Final Route Completion Audit" in rendered
    assert "Do custom GPX" in rendered
    assert "x<br>y" in rendered
