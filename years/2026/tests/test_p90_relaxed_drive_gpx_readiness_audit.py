from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_relaxed_drive_gpx_readiness_audit as audit  # noqa: E402


def test_has_stored_track_geometry_requires_segments():
    assert audit.has_stored_track_geometry(None) is False
    assert audit.has_stored_track_geometry({"trailhead_access": {"outbound_path_coordinates": [[0, 0]]}}) is False
    assert audit.has_stored_track_geometry({"segments": [{"seg_id": 1}]}) is True


def test_loop_rows_classifies_sources():
    draft = {
        "field_days": [
            {
                "draft_day_number": 1,
                "loops": [
                    {"loop_id": "p", "source": "personal_route_menu", "candidate_id": "p", "label": "P", "trailhead": "T"},
                    {"loop_id": "c", "source": "canonical_field_menu", "candidate_id": "c", "label": "C", "trailhead": "T"},
                    {"loop_id": "f", "source": "forced_anchor_probe", "candidate_id": "f", "label": "F", "trailhead": "T"},
                ],
            }
        ]
    }
    personal = {"p": {"segments": [{"seg_id": 1}]}}

    rows = audit.loop_rows(
        draft,
        personal_candidates=personal,
        hybrid_candidates={},
        field_packet_gpx={},
        forced_anchor_gpx={},
    )

    assert [row["readiness"] for row in rows] == [
        "stored_geometry_exportable",
        "needs_field_packet_gpx_lookup",
        "needs_probe_regeneration_for_coordinates",
    ]


def test_loop_rows_resolves_canonical_field_packet_gpx():
    draft = {
        "field_days": [
            {
                "draft_day_number": 1,
                "loops": [
                    {"loop_id": "c", "source": "canonical_field_menu", "candidate_id": "c", "label": "C", "trailhead": "T"},
                ],
            }
        ]
    }
    field_packet = {
        "c": {
            "gpx_path": "docs/field-packet/gpx/official/c.gpx",
            "gpx_exists": True,
            "validation_passed": True,
        }
    }

    rows = audit.loop_rows(
        draft,
        personal_candidates={},
        hybrid_candidates={},
        field_packet_gpx=field_packet,
        forced_anchor_gpx={},
    )

    assert rows[0]["readiness"] == "existing_navigation_gpx_available"
    assert rows[0]["existing_gpx_validation_passed"] is True


def test_loop_rows_resolves_generated_forced_anchor_gpx():
    draft = {
        "field_days": [
            {
                "draft_day_number": 1,
                "loops": [
                    {"loop_id": "forced-loop", "source": "forced_anchor_probe", "candidate_id": "f", "label": "F", "trailhead": "T"},
                ],
            }
        ]
    }
    forced = {"forced-loop": {"path": "forced.gpx", "validation_passed": True}}

    rows = audit.loop_rows(
        draft,
        personal_candidates={},
        hybrid_candidates={},
        field_packet_gpx={},
        forced_anchor_gpx=forced,
    )

    assert rows[0]["readiness"] == "generated_forced_anchor_gpx_available"
    assert rows[0]["forced_anchor_gpx_path"] == "forced.gpx"


def test_build_report_marks_day_level_ready_from_manifest():
    draft = {
        "field_days": [
            {
                "draft_day_number": 1,
                "loops": [
                    {"loop_id": "p", "source": "personal_route_menu", "candidate_id": "p", "label": "P", "trailhead": "T"},
                ],
            }
        ]
    }

    report = audit.build_report(
        draft,
        personal_candidates={"p": {"segments": [{"seg_id": 1}]}},
        hybrid_candidates={},
        field_packet_gpx={},
        forced_anchor_gpx={},
        day_level_gpx_manifest={
            "summary": {
                "day_gpx_count": 31,
                "loop_validation_passed": True,
                "day_track_validation_passed": True,
                "failed_day_count": 0,
            }
        },
    )

    assert report["summary"]["day_level_gpx_ready"] is True
    assert report["summary"]["day_level_gpx_failed_day_count"] == 0
