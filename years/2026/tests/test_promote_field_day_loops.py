from pathlib import Path
import sys
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import promote_field_day_loops as promote  # noqa: E402


class NoAcceptedReplacements:
    def match_for_loop(self, _loop, _candidate):
        return None

    def blocking_match_for_loop(self, _loop):
        return None


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


def test_unique_label_for_loop_avoids_existing_label_collision():
    loop = {"draft_day_number": 22}

    assert promote.unique_label_for_loop(loop, "FD22A", {"FD22A"}) == "FD22B"


def test_replacement_endpoint_distance_respects_ascent_start_endpoint():
    official_index = {
        1: {
            "seg_id": 1,
            "direction": "ascent",
            "coordinates": [
                [-116.0, 43.0],
                [-116.0, 43.1],
            ],
        }
    }

    result = promote.route_endpoint_distance_for_replacement(
        record={"target_segment_ids": ["1"]},
        anchor={"lon": -116.0, "lat": 43.09},
        official_index=official_index,
    )

    assert result["credit_endpoint_used"] == "segment_start"
    assert result["direction_rule_checked"] is True
    assert result["anchor_to_credit_endpoint_distance_miles"] > 5


def test_reanchored_replacement_cue_is_marked_regenerated():
    cue = {}
    promote.apply_replacement_metadata_to_cue(
        cue,
        {
            "replacement_id": "fd14d-lower",
            "status": "active",
            "accepted_anchor_ref": "strava-parking-anchor-13",
            "route_card_status": "provisional_re_anchored",
            "packet_visibility": "visible_with_provisional_badge",
            "certified_route_card": False,
            "requires_field_walkthrough": True,
        },
        endpoint={"anchor_to_credit_endpoint_distance_miles": 0.01},
    )

    assert cue["accepted_replacement_id"] == "fd14d-lower"
    assert cue["cue_generation_mode"] == "regenerated_for_reanchored_candidate"
    assert cue["certified_route_card"] is False


def test_accepted_replacement_street_probe_anchor_is_rebuilt_from_connector_geojson():
    connector_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "Name": "West Hidden Springs Drive",
                    "source": "openstreetmap",
                    "highway": "residential",
                    "surface": "asphalt",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [-116.0001, 43.0001],
                        [-116.0002, 43.0002],
                    ],
                },
            }
        ],
    }
    official_segments_by_id = {
        1494: {
            "seg_id": 1494,
            "start": (-116.0002, 43.0002),
            "end": (-116.0003, 43.0003),
            "coordinates": [
                [-116.0002, 43.0002],
                [-116.0003, 43.0003],
            ],
        }
    }

    anchors = promote.accepted_replacement_street_probe_anchors(
        [
            {
                "accepted_anchor_ref": "street-probe-west-hidden-springs-drive",
                "target_segment_ids": ["1494"],
            }
        ],
        connector_geojson,
        official_segments_by_id,
    )

    assert [anchor["anchor_id"] for anchor in anchors] == ["street-probe-west-hidden-springs-drive"]


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


def test_canonical_field_menu_prefers_stored_candidate_over_stale_packet(monkeypatch):
    stored_candidate = {
        "candidate_id": "block-upper_8th_corrals_sidewinder",
        "trail_names": ["8th Street Motorcycle Trail", "Sidewinder Trail", "Corrals Trail"],
        "segment_ids": [1483, 1484, 1660, 1524],
        "segments": [{"seg_id": 1483}],
    }
    context = promote.PromotionContext.__new__(promote.PromotionContext)
    context.personal_candidates = {}
    context.hybrid_candidates = {"block-upper_8th_corrals_sidewinder": stored_candidate}
    context.official_index = {}
    context.connector_graph = {}
    context.base_map_data = {
        "route_cues": {
            "block-upper_8th_corrals_sidewinder": {
                "segments": [{"seg_id": 1576}, {"seg_id": 1577}],
                "title": "Stale Highlands packet cue",
            }
        }
    }

    def fail_if_packet_gpx_is_used(*_args, **_kwargs):
        raise AssertionError("canonical field-menu loop reused field-packet navigation GPX")

    monkeypatch.setattr(promote.day_gpx, "loop_track_segments", fail_if_packet_gpx_is_used)
    monkeypatch.setattr(
        promote.day_gpx,
        "candidate_track",
        lambda *_args, **_kwargs: [[(-116.0, 43.0), (-116.0, 43.0001)]],
    )

    loop = {
        "source": "canonical_field_menu",
        "candidate_id": "block-upper_8th_corrals_sidewinder",
    }

    assert context.candidate_for_loop(loop)["segment_ids"] == [1576, 1577]
    tracks, source = context.track_segments_for_loop(loop)
    assert tracks == [[(-116.0, 43.0), (-116.0, 43.0001)]]
    assert source == "stored_hybrid_candidate_from_canonical_field_menu"


def test_field_menu_replacement_index_matches_subset_partition():
    index = promote.FieldMenuReplacementIndex(
        {
            "overrides": [
                {
                    "reason": "accepted split",
                    "replace_package": {
                        "components": [
                            {"candidate_id": "dropped", "segment_ids": [99]},
                            {"candidate_id": "left", "segment_ids": [1, 2]},
                            {"candidate_id": "right", "segment_ids": [3]},
                        ]
                    },
                    "route_cues": {
                        "left": {"candidate_id": "left"},
                        "right": {"candidate_id": "right"},
                    },
                    "feature_collections": {
                        "routes": {
                            "features": [
                                {
                                    "type": "Feature",
                                    "properties": {"candidate_id": "left"},
                                    "geometry": {"type": "LineString", "coordinates": [[0, 0], [0, 0.01]]},
                                },
                                {
                                    "type": "Feature",
                                    "properties": {"candidate_id": "right"},
                                    "geometry": {"type": "LineString", "coordinates": [[0, 0.01], [0, 0.02]]},
                                },
                            ]
                        }
                    },
                }
            ]
        }
    )

    split = index.split_for_segment_ids({1, 2, 3})

    assert [component["candidate_id"] for component in split["components"]] == ["left", "right"]


def test_build_promoted_map_data_splits_selected_loop_from_field_menu_replacement(tmp_path):
    replacement_index = promote.FieldMenuReplacementIndex(
        {
            "overrides": [
                {
                    "reason": "accepted field executable split",
                    "remove_candidate_ids": ["collapsed"],
                    "replace_package": {
                        "components": [
                            {
                                "candidate_id": "west",
                                "trail_names": ["West Trail"],
                                "official_miles": 1.0,
                                "on_foot_miles": 1.2,
                                "total_minutes": 30,
                                "trailhead": "West Trailhead",
                                "segment_ids": [1, 2],
                            },
                            {
                                "candidate_id": "east",
                                "trail_names": ["East Trail"],
                                "official_miles": 1.5,
                                "on_foot_miles": 1.8,
                                "total_minutes": 45,
                                "trailhead": "East Trailhead",
                                "segment_ids": [3, 4],
                            },
                        ]
                    },
                    "route_cues": {
                        "west": {"candidate_id": "west", "segments": [{"seg_id": 1}, {"seg_id": 2}]},
                        "east": {"candidate_id": "east", "segments": [{"seg_id": 3}, {"seg_id": 4}]},
                    },
                    "feature_collections": {
                        "routes": {
                            "features": [
                                {
                                    "type": "Feature",
                                    "properties": {"candidate_id": "west"},
                                    "geometry": {"type": "LineString", "coordinates": [[0, 0], [0, 0.01]]},
                                },
                                {
                                    "type": "Feature",
                                    "properties": {"candidate_id": "east"},
                                    "geometry": {"type": "LineString", "coordinates": [[0, 0.02], [0, 0.03]]},
                                },
                            ]
                        },
                        "parking": {"features": []},
                    },
                }
            ]
        }
    )
    calendar = {
        "assignments": [
            {
                "date": "2026-07-06",
                "field_day": {
                    "draft_day_number": 12,
                    "field_day_id": "weekday-collapsed",
                    "loops": [
                        {
                            "loop_id": "hybrid_candidate_index::collapsed::Trailhead",
                            "source": "hybrid_candidate_index",
                            "candidate_id": "collapsed",
                            "trailhead": "Trailhead",
                            "trail_names": ["Collapsed"],
                        }
                    ],
                },
            }
        ]
    }
    candidate = {"candidate_id": "collapsed", "segment_ids": [1, 2, 3, 4]}
    context = SimpleNamespace(
        args=SimpleNamespace(
            field_packet_dir=tmp_path,
            calendar_json=tmp_path / "calendar.json",
            base_map_data_json=tmp_path / "base.json",
            field_tool_json=tmp_path / "field-tool-data.json",
            accepted_replacements_json=tmp_path / "accepted.json",
            segment_promotions_json=tmp_path / "promotions.json",
            field_menu_replacements_json=tmp_path / "replacements.json",
            personal_route_menu_json=tmp_path / "personal.json",
            hybrid_route_pass_json=tmp_path / "hybrid.json",
            forced_anchor_probe_json=tmp_path / "forced.json",
        ),
        accepted_replacements=NoAcceptedReplacements(),
        field_menu_replacements=replacement_index,
        official_segments=[
            {"seg_id": 1, "official_miles": 0.5, "coordinates": [[0, 0], [0, 0.005]]},
            {"seg_id": 2, "official_miles": 0.5, "coordinates": [[0, 0.005], [0, 0.01]]},
            {"seg_id": 3, "official_miles": 0.75, "coordinates": [[0, 0.02], [0, 0.025]]},
            {"seg_id": 4, "official_miles": 0.75, "coordinates": [[0, 0.025], [0, 0.03]]},
        ],
        candidate_for_loop=lambda _loop: candidate,
    )

    map_data, report = promote.build_promoted_map_data(
        calendar=calendar,
        base_map_data={"progress": {}},
        field_tool_payload={"routes": []},
        segment_promotions_payload={"promotions": []},
        context=context,
    )

    package = map_data["packages"][0]
    assert package["package_number"] == 112
    assert package["component_candidate_ids"] == ["west", "east"]
    assert [component["field_menu_label"] for component in package["components"]] == ["FD12A", "FD12B"]
    assert [component["field_menu_group_id"] for component in package["components"]] == [
        "hybrid_candidate_index::collapsed::Trailhead::west",
        "hybrid_candidate_index::collapsed::Trailhead::east",
    ]
    assert package["segment_ids"] == [1, 2, 3, 4]
    assert {row["mode"] for row in report["promotions"]} == {"split_selected_loop_via_field_menu_replacement"}


def test_build_promoted_map_data_does_not_apply_unscoped_field_menu_replacement(tmp_path):
    replacement_index = promote.FieldMenuReplacementIndex(
        {
            "overrides": [
                {
                    "reason": "replacement for another source candidate",
                    "remove_candidate_ids": ["different-old-candidate"],
                    "replace_package": {
                        "components": [
                            {
                                "candidate_id": "west",
                                "trail_names": ["West Trail"],
                                "official_miles": 1.0,
                                "on_foot_miles": 1.2,
                                "total_minutes": 30,
                                "trailhead": "West Trailhead",
                                "segment_ids": [1, 2],
                            },
                            {
                                "candidate_id": "east",
                                "trail_names": ["East Trail"],
                                "official_miles": 1.5,
                                "on_foot_miles": 1.8,
                                "total_minutes": 45,
                                "trailhead": "East Trailhead",
                                "segment_ids": [3, 4],
                            },
                        ]
                    },
                    "route_cues": {
                        "west": {"candidate_id": "west", "segments": [{"seg_id": 1}, {"seg_id": 2}]},
                        "east": {"candidate_id": "east", "segments": [{"seg_id": 3}, {"seg_id": 4}]},
                    },
                    "feature_collections": {
                        "routes": {
                            "features": [
                                {
                                    "type": "Feature",
                                    "properties": {"candidate_id": "west"},
                                    "geometry": {"type": "LineString", "coordinates": [[0, 0], [0, 0.01]]},
                                },
                                {
                                    "type": "Feature",
                                    "properties": {"candidate_id": "east"},
                                    "geometry": {"type": "LineString", "coordinates": [[0, 0.02], [0, 0.03]]},
                                },
                            ]
                        },
                        "parking": {"features": []},
                    },
                }
            ]
        }
    )
    calendar = {
        "assignments": [
            {
                "date": "2026-07-06",
                "field_day": {
                    "draft_day_number": 12,
                    "field_day_id": "weekday-collapsed",
                    "loops": [
                        {
                            "loop_id": "hybrid_candidate_index::collapsed::Trailhead",
                            "source": "hybrid_candidate_index",
                            "candidate_id": "collapsed",
                            "trailhead": "Trailhead",
                            "trail_names": ["Collapsed"],
                            "official_miles": 2.5,
                            "on_foot_miles": 2.8,
                            "p75_minutes": 60,
                        }
                    ],
                },
            }
        ]
    }
    candidate = {
        "candidate_id": "collapsed",
        "trail_names": ["Collapsed"],
        "trailhead": {"name": "Trailhead"},
        "segment_ids": [1, 2, 3, 4],
        "segments": [{"seg_id": 1, "trail_name": "Collapsed"}],
    }
    context = SimpleNamespace(
        args=SimpleNamespace(
            field_packet_dir=tmp_path,
            calendar_json=tmp_path / "calendar.json",
            base_map_data_json=tmp_path / "base.json",
            field_tool_json=tmp_path / "field-tool-data.json",
            accepted_replacements_json=tmp_path / "accepted.json",
            segment_promotions_json=tmp_path / "promotions.json",
            field_menu_replacements_json=tmp_path / "replacements.json",
            personal_route_menu_json=tmp_path / "personal.json",
            hybrid_route_pass_json=tmp_path / "hybrid.json",
            forced_anchor_probe_json=tmp_path / "forced.json",
        ),
        accepted_replacements=NoAcceptedReplacements(),
        field_menu_replacements=replacement_index,
        official_segments=[
            {"seg_id": 1, "official_miles": 0.5, "coordinates": [[0, 0], [0, 0.005]]},
            {"seg_id": 2, "official_miles": 0.5, "coordinates": [[0, 0.005], [0, 0.01]]},
            {"seg_id": 3, "official_miles": 0.75, "coordinates": [[0, 0.02], [0, 0.025]]},
            {"seg_id": 4, "official_miles": 0.75, "coordinates": [[0, 0.025], [0, 0.03]]},
        ],
        candidate_for_loop=lambda _loop: candidate,
        track_segments_for_loop=lambda _loop: ([[[0, 0], [0, 0.03]]], "test_track"),
    )

    map_data, report = promote.build_promoted_map_data(
        calendar=calendar,
        base_map_data={"progress": {}},
        field_tool_payload={"routes": []},
        segment_promotions_payload={"promotions": []},
        context=context,
    )

    package = map_data["packages"][0]
    assert package["component_candidate_ids"] == ["collapsed"]
    assert package["components"][0]["on_foot_miles"] == 2.8
    assert {row["mode"] for row in report["promotions"]} == {"promoted_candidate_to_route_card_source"}
