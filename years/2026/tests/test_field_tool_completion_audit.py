import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "field_tool_completion_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("field_tool_completion_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_audit_inputs(tmp_path):
    module = load_module()
    canonical = {"summary": {"component_route_count": 1}, "packages": [{"package_number": 1}]}
    packet_dir = tmp_path / "packet"
    nav_dir = packet_dir / "gpx" / "navigation"
    nav_dir.mkdir(parents=True)
    (nav_dir / "route.gpx").write_text("<gpx />\n", encoding="utf-8")
    (packet_dir / "index.html").write_text(
        """
        <button data-filter="60"></button><button data-filter="90"></button>
        <button data-filter="120"></button><button data-filter="180"></button>
        <button data-filter="240"></button><button data-filter="360"></button>
        <button>Mark done</button><button>Hide completed</button><button>Export progress</button>
        <script>
        const x = "fieldPacketCompletedOutings";
        function completedSegmentSet() {}
        const note = "missed_segment_ids";
        const best = "Best today for 120 minutes: route · 4 new official segment(s) · completion-safe in the current menu";
        </script>
        """,
        encoding="utf-8",
    )
    for name in ["field-tool-data.json", "manifest.json", "service-worker.js"]:
        (packet_dir / name).write_text("{}", encoding="utf-8")
    field_tool_data = {
        "schema": "boise_trails_field_tool_data_v1",
        "source": {"map_data_sha256": module.stable_json_sha256(canonical)},
        "time_filters_minutes": [60, 90, 120, 180, 240, 360],
        "certified_baseline": {
            "status": "passed",
            "official_segment_count": 251,
            "covered_segment_count": 251,
            "missing_segment_count": 0,
        },
        "summary": {"segment_count_in_field_menu": 1},
        "routes": [
            {
                "outing_id": "1-1",
                "label": "1A",
                "trailhead": "Trailhead",
                "segment_ids": ["101"],
                "door_to_door_minutes_p75": 75,
                "door_to_door_minutes_p90": 90,
                "official_miles": 1.0,
                "on_foot_miles": 1.4,
                "gpx_href": "gpx/navigation/route.gpx",
                "parking": {"lat": 43.1, "lon": -116.1, "has_parking": True},
                "effort": {
                    "ascent_ft": 120,
                    "descent_ft": 80,
                    "grade_adjusted_miles": 1.12,
                    "estimated_moving_minutes_p75": 35,
                },
                "validation": {"passed": True},
                "completion_safety": {
                    "normal_completion_preserves_remaining_menu_coverage": True,
                    "missing_remaining_segment_ids_after_completion": [],
                },
                "turn_by_turn_steps": [
                    {"kind": "park"},
                    {"kind": "navigate"},
                    {"kind": "return"},
                ],
            }
        ],
    }
    manifest = {
        "summary": {
            "gpx_validation_passed": True,
            "navigation_gpx_count": 1,
            "failed_gpx_count": 0,
        }
    }
    official_geojson = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"segId": 101}, "geometry": None}],
    }
    recertification = {
        "status": "passed",
        "summary": {
            "remaining_full_completion_feasible": True,
            "remaining_coverage_preserved": True,
        },
    }
    return {
        "field_tool_data": field_tool_data,
        "manifest": manifest,
        "official_geojson": official_geojson,
        "index_html": (packet_dir / "index.html").read_text(encoding="utf-8"),
        "packet_dir": packet_dir,
        "canonical_map_data": canonical,
        "recertification_report": recertification,
    }


def test_completion_audit_passes_when_field_tool_contract_is_met(tmp_path):
    module = load_module()
    audit = module.build_completion_audit(**sample_audit_inputs(tmp_path))

    assert audit["status"] == "passed"
    assert audit["summary"]["passed_requirement_count"] == audit["summary"]["requirement_count"]


def test_completion_audit_fails_when_canonical_source_hash_does_not_match(tmp_path):
    module = load_module()
    inputs = sample_audit_inputs(tmp_path)
    inputs["field_tool_data"]["source"]["map_data_sha256"] = "stale"

    audit = module.build_completion_audit(**inputs)

    assert audit["status"] == "failed"
    checks = {check["requirement"]: check for check in audit["checks"]}
    assert checks["Phone page and map share the canonical field-menu source"]["passed"] is False
