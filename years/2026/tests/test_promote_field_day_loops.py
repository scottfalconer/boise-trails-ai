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


def test_enrich_official_repeat_segment_ids_from_route_geometry():
    map_data = {
        "feature_collections": {
            "routes": {
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"candidate_id": "route-a"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [-116.0, 43.0],
                                [-116.0, 43.01],
                                [-116.0, 43.02],
                                [-116.0, 43.03],
                            ],
                        },
                    }
                ]
            },
            "official_segments": {
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "seg_id": 101,
                            "segment_name": "Connector Trail 1",
                            "trail_name": "Connector Trail",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-116.0, 43.01], [-116.0, 43.02]],
                        },
                    }
                ]
            },
        },
        "route_cues": {
            "route-a": {
                "segments": [
                    {"seg_id": 201, "trail_name": "First Trail", "official_miles": 0.2},
                    {"seg_id": 202, "trail_name": "Second Trail", "official_miles": 0.2},
                ],
                "between_links": [
                    {
                        "distance_miles": 0.5,
                        "official_repeat_miles": 0.1,
                        "connector_names": ["Connector Trail"],
                        "connector_classes": ["official_repeat"],
                    }
                ],
            }
        },
    }

    promote.enrich_official_repeat_segment_ids(map_data)

    assert map_data["route_cues"]["route-a"]["between_links"][0]["official_repeat_segment_ids"] == [101]


def test_enrich_official_repeat_segment_ids_from_geometry_without_names():
    map_data = {
        "feature_collections": {
            "routes": {
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"candidate_id": "route-a"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-116.0, 43.0], [-116.0, 43.01], [-116.0, 43.02]],
                        },
                    }
                ]
            },
            "official_segments": {
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"seg_id": 111, "segment_name": "Repeat Trail 1", "trail_name": "Repeat Trail"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-116.0, 43.01], [-116.0, 43.02]],
                        },
                    }
                ]
            },
        },
        "route_cues": {
            "route-a": {
                "segments": [{"seg_id": 201, "trail_name": "First Trail", "official_miles": 0.1}],
                "return_to_car": {
                    "strategy": "accepted_manual_split_gpx",
                    "official_repeat_miles": 0.2,
                    "connector_miles": 0.0,
                    "connector_names": [],
                    "connector_classes": [],
                },
            }
        },
    }

    promote.enrich_official_repeat_segment_ids(map_data)

    assert map_data["route_cues"]["route-a"]["return_to_car"]["official_repeat_segment_ids"] == [111]


def test_enrich_out_and_back_return_uses_claimed_segment_ids():
    map_data = {
        "feature_collections": {
            "routes": {
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"candidate_id": "route-a"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-116.0, 43.0], [-116.0, 43.01]],
                        },
                    }
                ]
            },
            "official_segments": {"features": []},
        },
        "route_cues": {
            "route-a": {
                "segments": [{"seg_id": 301, "trail_name": "Out Trail", "official_miles": 0.5}],
                "return_to_car": {
                    "strategy": "out_and_back",
                    "official_repeat_miles": 0.5,
                    "connector_names": [],
                    "connector_classes": [],
                },
            }
        },
    }

    promote.enrich_official_repeat_segment_ids(map_data)

    assert map_data["route_cues"]["route-a"]["return_to_car"]["official_repeat_segment_ids"] == [301]


def test_segment_ownership_promotion_updates_component_and_cue():
    component = {
        "candidate_id": "chbh-connector",
        "trail_names": ["CHBH Connector"],
        "official_miles": 0.81,
        "on_foot_miles": 3.16,
        "segment_ids": [1516],
    }
    cue = {
        "candidate_id": "chbh-connector",
        "segments": [{"seg_id": 1516, "trail_name": "CHBH Connector", "official_miles": 0.81}],
        "start_access": {"official_repeat_segment_ids": [1541, 1610]},
    }
    promotions = [
        {
            "segment_id": 1610,
            "reason": "Quick Draw is already covered by FD14B.",
            "from": {"candidate_id": "quick-draw"},
            "to": {"candidate_id": "chbh-connector", "insert_after_segment_id": 1516},
        }
    ]
    base_map_data = {
        "route_cues": {
            "quick-draw": {
                "segments": [{"seg_id": 1610, "trail_name": "Quick Draw", "official_miles": 0.48}]
            }
        }
    }

    promote.apply_segment_promotions_to_route_card(
        component=component,
        cue=cue,
        promotions=promotions,
        base_map_data=base_map_data,
        official_by_id={"1610": {"seg_id": 1610, "trail_name": "Quick Draw", "official_miles": 0.48}},
    )

    assert component["segment_ids"] == [1516, 1610]
    assert component["official_miles"] == 1.29
    assert component["trail_names"] == ["CHBH Connector", "Quick Draw"]
    assert [segment["seg_id"] for segment in cue["segments"]] == [1516, 1610]
    assert cue["start_access"]["official_repeat_segment_ids"] == [1541]


def test_removed_source_loop_targets_collect_source_action_removals():
    rows = promote.removed_source_loop_targets(
        {
            "promotions": [
                {
                    "status": "promoted",
                    "segment_id": 1576,
                    "source_action": "remove_route_card",
                    "reason": "Highlands is covered by route 12.",
                    "from": {"candidate_id": "highlands-trail"},
                    "to": {"candidate_id": "block-upper_8th_corrals_sidewinder"},
                },
                {
                    "status": "promoted",
                    "segment_id": 1656,
                    "from": {"candidate_id": "manual-16a-2"},
                    "to": {"candidate_id": "dry-creek"},
                },
            ]
        }
    )

    assert rows == {
        "highlands-trail": {
            "source_candidate_id": "highlands-trail",
            "target_candidate_id": "block-upper_8th_corrals_sidewinder",
            "segment_ids": ["1576"],
            "reasons": ["Highlands is covered by route 12."],
        }
    }


def test_forced_probe_index_matches_trailhead_suffixed_candidate_ids():
    rows = promote.forced_probe_index(
        {
            "probe_rows": [
                {
                    "candidate_id": "single-segment-1661-spring-creek",
                    "anchor_name": "Avimor Spring Valley Creek parking",
                    "parking_confidence": "osm_amenity_parking_fee_no_capacity_36_source_checked",
                }
            ]
        }
    )

    assert rows[
        (
            "single-segment-1661-spring-creek::Avimor Spring Valley Creek parking",
            "Avimor Spring Valley Creek parking",
        )
    ]["parking_confidence"] == "osm_amenity_parking_fee_no_capacity_36_source_checked"


def test_sync_missing_cue_trailhead_metadata_uses_candidate_source():
    cue = {"trailhead": {"name": "Avimor Spring Valley Creek parking", "parking_confidence": None}}
    candidate = {
        "trailhead": {
            "parking_confidence": "osm_amenity_parking_fee_no_capacity_36_source_checked",
            "source": "osm_overpass_amenity_parking_2026_05_06_plus_alltrails_spring_valley_creek",
            "field_ready": True,
        }
    }

    promote.sync_missing_cue_trailhead_metadata(cue, candidate)

    assert cue["trailhead"]["parking_confidence"] == "osm_amenity_parking_fee_no_capacity_36_source_checked"
    assert cue["trailhead"]["source"] == "osm_overpass_amenity_parking_2026_05_06_plus_alltrails_spring_valley_creek"
    assert cue["trailhead"]["field_ready"] is True
