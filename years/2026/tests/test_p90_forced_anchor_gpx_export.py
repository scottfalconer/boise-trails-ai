from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_forced_anchor_gpx_export as export  # noqa: E402


def test_forced_anchor_loops_extracts_only_forced_rows():
    draft = {
        "field_days": [
            {
                "draft_day_number": 7,
                "loops": [
                    {"source": "forced_anchor_probe", "loop_id": "forced"},
                    {"source": "personal_route_menu", "loop_id": "personal"},
                ],
            }
        ]
    }

    loops = export.forced_anchor_loops(draft)

    assert loops == [{"source": "forced_anchor_probe", "loop_id": "forced", "draft_day_number": 7}]


def test_segment_id_from_candidate_id_parses_probe_id():
    assert export.segment_id_from_candidate_id("single-segment-1661-spring-creek::Anchor") == 1661


def test_anchor_by_name_finds_exact_name():
    anchor = {"name": "A"}

    assert export.anchor_by_name([anchor], "A") is anchor
