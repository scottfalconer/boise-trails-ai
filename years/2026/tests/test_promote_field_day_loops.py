from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import promote_field_day_loops as promote  # noqa: E402


def test_route_component_uses_loop_id_as_group_boundary():
    loop = {
        "loop_id": "personal_route_menu::kestral-trail::Hulls Gulch Trailhead",
        "source": "personal_route_menu",
        "candidate_id": "kestral-trail",
        "trailhead": "Hulls Gulch Trailhead",
        "trail_names": ["Kestral Trail"],
        "official_miles": 0.75,
        "on_foot_miles": 1.55,
        "p75_minutes": 55,
        "p90_minutes": 62,
        "promotion_route_number": 1,
    }
    candidate = {
        "candidate_id": "kestral-trail",
        "trail_names": ["Kestral Trail"],
        "segment_ids": [1583],
        "route_status": "graph_validated",
        "time_estimates_minutes": {"door_to_door_p75": 55, "door_to_door_p90": 62},
    }

    component = promote.route_component(
        loop=loop,
        candidate_id="kestral-trail",
        candidate=candidate,
        existing_route=None,
        label="FD19A",
    )

    assert component["field_menu_group_id"] == loop["loop_id"]
    assert component["candidate_id"] == "kestral-trail"
    assert component["field_menu_label"] == "FD19A"
    assert component["segment_ids"] == [1583]
    assert component["time_estimates_minutes"]["door_to_door_p90"] == 62


def test_existing_certified_route_label_is_preserved():
    loop = {"draft_day_number": 16}
    existing = {"label": "16A-2"}

    assert promote.label_for_loop(loop, existing, 0) == "16A-2"


def test_package_for_day_keeps_same_trailhead_loops_separate():
    day = {"draft_day_number": 19, "field_day_id": "weekday-a", "date": "2026-06-18"}
    components = [
        {
            "candidate_id": "kestral-trail",
            "field_menu_group_id": "personal_route_menu::kestral-trail::Hulls Gulch Trailhead",
            "trailhead": "Hulls Gulch Trailhead",
            "trail_names": ["Kestral Trail"],
            "official_miles": 0.75,
            "on_foot_miles": 1.55,
            "total_minutes": 55,
            "segment_ids": [1583],
        },
        {
            "candidate_id": "lower-hulls-gulch-trail-red-cliffs",
            "field_menu_group_id": "personal_route_menu::lower-hulls-gulch-trail-red-cliffs::Hulls Gulch Trailhead",
            "trailhead": "Hulls Gulch Trailhead",
            "trail_names": ["Lower Hull's Gulch Trail", "Red Cliffs"],
            "official_miles": 3.45,
            "on_foot_miles": 4.92,
            "total_minutes": 104,
            "segment_ids": [1585, 1586, 1587, 1588, 1589, 1615, 1616],
        },
    ]

    package = promote.package_for_day(day, components)

    assert package["package_number"] == 119
    assert package["component_candidate_ids"] == [
        "kestral-trail",
        "lower-hulls-gulch-trail-red-cliffs",
    ]
    assert [component["field_menu_group_id"] for component in package["components"]] == [
        "personal_route_menu::kestral-trail::Hulls Gulch Trailhead",
        "personal_route_menu::lower-hulls-gulch-trail-red-cliffs::Hulls Gulch Trailhead",
    ]
