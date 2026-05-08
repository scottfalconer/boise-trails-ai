from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multi_start_alternative_audit as audit  # noqa: E402


def test_merge_parking_anchors_dedupes_and_keeps_stronger_confidence():
    weaker = {
        "anchor_id": "public-1",
        "name": "Same Lot",
        "lat": 43.1,
        "lon": -116.1,
        "parking_confidence": "inferred_from_trailhead_layer",
        "source": "city_parks_facilities",
        "has_parking": True,
    }
    stronger = {
        "anchor_id": "manual-1",
        "name": "Same Lot",
        "lat": 43.1000002,
        "lon": -116.1000002,
        "parking_confidence": "source_verified_roadside_plus_strava_seen",
        "source": "manual_source_check",
        "has_parking": True,
    }

    merged = audit.merge_parking_anchors([weaker, stronger])

    assert len(merged) == 1
    assert merged[0]["anchor_id"] == "manual-1"
    assert merged[0]["parking_confidence"] == "source_verified_roadside_plus_strava_seen"


def test_street_parking_probes_assume_paved_public_vehicle_roads_within_point_one_mile():
    segment = {
        "seg_id": 1,
        "start": (-116.1001, 43.1001),
        "end": (-116.101, 43.101),
        "center": (-116.1005, 43.1005),
    }
    connector_geojson = {
        "type": "FeatureCollection",
        "features": [
            road_feature("North Good Street", "residential", None, -116.1002, 43.1002),
            road_feature("OSM service connector 99", "service", "private", -116.1003, 43.1003),
            road_feature("Trail-ish Footway", "footway", None, -116.1004, 43.1004),
            road_feature("Fast Road", "primary", None, -116.1005, 43.1005),
            road_feature("Shoulder Road", "secondary", None, -116.1007, 43.1007),
            road_feature("Public Tertiary", "tertiary", None, -116.1006, 43.1006),
            road_feature("Paved Service Road", "service", None, -116.1008, 43.1008, surface="asphalt"),
            road_feature("Unknown Service Road", "service", None, -116.1009, 43.1009),
            road_feature("Gravel Residential Road", "residential", None, -116.1002, 43.1002, surface="gravel"),
            road_feature("Gravel Track", "track", None, -116.1002, 43.1002, surface="gravel"),
            road_feature("Too Far Residential", "residential", None, -116.103, 43.103),
            road_feature("Mid Segment Road", "residential", None, -116.0955, 43.0955),
        ],
    }

    probes = audit.street_parking_probes_for_segments(
        connector_geojson,
        [segment],
        limit=10,
    )

    names = sorted(probe["name"] for probe in probes)
    assert names == sorted([
        "North Good Street road-parking anchor",
        "Fast Road road-parking anchor",
        "Paved Service Road road-parking anchor",
        "Public Tertiary road-parking anchor",
        "Shoulder Road road-parking anchor",
    ])
    assert all(
        probe["parking_confidence"] == audit.ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE
        for probe in probes
    )
    assert all(probe["field_ready"] is False for probe in probes)
    assert all(probe["max_assumed_walk_to_road_miles"] == 0.1 for probe in probes)


def test_ranked_anchors_limits_assumed_road_parking_to_component_distance():
    segment = {
        "seg_id": 1,
        "start": (-116.1001, 43.1001),
        "end": (-116.101, 43.101),
        "center": (-116.1005, 43.1005),
    }
    near = {
        "anchor_id": "near-road",
        "name": "Near Road",
        "lat": 43.1002,
        "lon": -116.1002,
        "parking_confidence": audit.ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE,
        "source": "openstreetmap_public_road_probe",
        "source_type": "assumed_vehicle_road_parking",
        "has_parking": True,
    }
    far = {
        "anchor_id": "far-road",
        "name": "Far Road",
        "lat": 43.103,
        "lon": -116.103,
        "parking_confidence": audit.ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE,
        "source": "openstreetmap_public_road_probe",
        "source_type": "assumed_vehicle_road_parking",
        "has_parking": True,
    }

    ranked = audit.ranked_anchors_for_component([near, far], [segment], limit=10)

    assert [anchor["anchor_id"] for anchor in ranked] == ["near-road"]


def test_street_parking_probes_do_not_use_segment_center_as_access():
    segment = {
        "seg_id": 1,
        "start": (-116.100, 43.100),
        "end": (-116.100, 43.110),
        "center": (-116.100, 43.105),
    }
    connector_geojson = {
        "type": "FeatureCollection",
        "features": [
            road_feature("Center Only Road", "residential", None, -116.101, 43.104),
            road_feature("Endpoint Road", "residential", None, -116.101, 43.099),
        ],
    }

    probes = audit.street_parking_probes_for_segments(
        connector_geojson,
        [segment],
        limit=10,
    )

    assert [probe["name"] for probe in probes] == ["Endpoint Road road-parking anchor"]


def test_ranked_anchors_does_not_treat_segment_center_as_road_parking_access():
    segment = {
        "seg_id": 1,
        "start": (-116.100, 43.100),
        "end": (-116.100, 43.110),
        "center": (-116.100, 43.105),
    }
    center_only = {
        "anchor_id": "center-road",
        "name": "Center Road",
        "lat": 43.105,
        "lon": -116.100,
        "parking_confidence": audit.ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE,
        "source": "openstreetmap_public_road_probe",
        "source_type": "assumed_vehicle_road_parking",
        "has_parking": True,
    }

    ranked = audit.ranked_anchors_for_component([center_only], [segment], limit=10)

    assert ranked == []


def test_field_ready_public_trailhead_inference_is_not_a_parking_blocker():
    blockers = audit.parking_blockers_for_anchor(
        {
            "anchor_id": "facility-808",
            "name": "Cartwright Trailhead",
            "parking_confidence": "inferred_from_trailhead_layer",
            "field_ready": True,
            "source_type": "public_trailhead",
            "has_parking": True,
        }
    )

    assert blockers == []


def test_assumed_paved_road_parking_stays_review_blocked():
    blockers = audit.parking_blockers_for_anchor(
        {
            "anchor_id": "street-probe-good-road",
            "name": "Good Road road-parking anchor",
            "parking_confidence": audit.ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE,
            "field_ready": False,
            "source_type": "assumed_vehicle_road_parking",
            "has_parking": True,
        }
    )

    assert "assumed paved-road parking requires manual check" in blockers
    assert "anchor is not field-ready" in blockers


def test_single_private_strava_anchor_is_valid_prior_parking_not_a_blocker():
    anchor = audit.normalized_anchor(
        {
            "anchor_id": "strava-parking-anchor-21",
            "name": "Strava parking anchor 21",
            "lat": 43.1,
            "lon": -116.1,
            "parking_confidence": "strava_single_prior_challenge_window",
            "source": "private_strava_anchor",
            "source_type": "private_strava_anchor",
            "has_parking": True,
            "privacy": "private_exact_coordinates",
        }
    )

    assert anchor["field_ready"] is True
    assert audit.parking_blockers_for_anchor(anchor) == []


def test_private_strava_anchor_source_is_valid_even_without_known_confidence_label():
    anchor = audit.normalized_anchor(
        {
            "anchor_id": "strava-parking-anchor-future",
            "name": "Strava parking anchor future",
            "lat": 43.1,
            "lon": -116.1,
            "parking_confidence": "future_strava_endpoint_label",
            "source": "strava_activity_endpoint_cluster",
            "source_type": "private_strava_anchor",
            "has_parking": True,
            "field_ready": False,
            "privacy": "private_exact_coordinates",
        }
    )

    assert anchor["field_ready"] is True
    assert audit.parking_blockers_for_anchor(anchor) == []


def test_parking_review_yes_confirms_assumed_paved_road_anchor():
    reviewed = audit.apply_parking_review_decisions(
        [
            {
                "anchor_id": "street-probe-good-road",
                "name": "Good Road road-parking anchor",
                "lat": 43.1,
                "lon": -116.1,
                "parking_confidence": audit.ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE,
                "field_ready": False,
                "source_type": "assumed_paved_road_parking",
                "has_parking": True,
            }
        ],
        {"street-probe-good-road": {"decision": "yes", "notes": "", "updatedAt": "2026-05-08T11:47:12Z"}},
    )[0]

    assert reviewed["field_ready"] is True
    assert reviewed["parking_confidence"] == "user_review_confirmed_paved_road_parking"
    assert audit.parking_blockers_for_anchor(reviewed) == []


def test_parking_review_maybe_keeps_assumed_paved_road_anchor_blocked():
    reviewed = audit.apply_parking_review_decisions(
        [
            {
                "anchor_id": "street-probe-good-road",
                "name": "Good Road road-parking anchor",
                "lat": 43.1,
                "lon": -116.1,
                "parking_confidence": audit.ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE,
                "field_ready": False,
                "source_type": "assumed_paved_road_parking",
                "has_parking": True,
            }
        ],
        {"street-probe-good-road": {"decision": "maybe", "notes": "", "updatedAt": "2026-05-08T11:47:12Z"}},
    )[0]

    blockers = audit.parking_blockers_for_anchor(reviewed)
    assert reviewed["field_ready"] is False
    assert "user marked parking maybe" in blockers
    assert "assumed paved-road parking requires manual check" in blockers


def test_parking_review_no_excludes_anchor_from_ranking():
    segment = {
        "seg_id": 1,
        "trail_name": "Neighborhood Trail",
        "start": (-116.1001, 43.1001),
        "end": (-116.101, 43.101),
        "center": (-116.1005, 43.1005),
    }
    anchors = audit.apply_parking_review_decisions(
        [
            {
                "anchor_id": "street-probe-rejected",
                "name": "Rejected Road road-parking anchor",
                "lat": 43.1002,
                "lon": -116.1002,
                "parking_confidence": audit.ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE,
                "field_ready": False,
                "source_type": "assumed_paved_road_parking",
                "has_parking": True,
            }
        ],
        {"street-probe-rejected": {"decision": "no", "notes": "", "updatedAt": "2026-05-08T11:47:12Z"}},
    )

    assert audit.ranked_anchors_for_component(anchors, [segment], limit=10) == []


def test_bogus_basin_components_only_rank_known_lodge_trailhead_anchors():
    segment = {
        "seg_id": 1,
        "trail_name": "Lodge Trail",
        "start": (-116.1001, 43.1001),
        "end": (-116.101, 43.101),
        "center": (-116.1005, 43.1005),
    }
    assumed_road = {
        "anchor_id": "near-road",
        "name": "Lodge Cat Track road-parking anchor",
        "lat": 43.1002,
        "lon": -116.1002,
        "parking_confidence": audit.ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE,
        "source": "openstreetmap_public_road_probe",
        "source_type": "assumed_vehicle_road_parking",
        "has_parking": True,
    }
    private_seen = {
        "anchor_id": "private-bogus",
        "name": "Strava parking anchor 16",
        "lat": 43.1003,
        "lon": -116.1003,
        "parking_confidence": "strava_seen_prior_challenge_window",
        "source": "private_strava_anchor",
        "source_type": "private_strava_anchor",
        "has_parking": True,
    }
    known_lodge = {
        "anchor_id": "facility-821",
        "name": "Pioneer Lodge Parking Area",
        "lat": 43.102,
        "lon": -116.102,
        "parking_confidence": "inferred_from_trailhead_layer",
        "source": "city_parks_facilities",
        "source_type": "public_trailhead",
        "has_parking": True,
        "field_ready": True,
    }

    ranked = audit.ranked_anchors_for_component([assumed_road, private_seen, known_lodge], [segment], limit=10)

    assert [anchor["anchor_id"] for anchor in ranked] == ["facility-821"]


def road_feature(name, highway, access, lon, lat, *, surface=None):
    props = {
        "Name": name,
        "TrailName": name,
        "source": "openstreetmap",
        "highway": highway,
    }
    if access is not None:
        props["access"] = access
    if surface is not None:
        props["surface"] = surface
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon, lat], [lon + 0.001, lat + 0.001]],
        },
    }


def test_classify_alternative_prefers_needs_parking_check_for_manual_good_math():
    classification = audit.classify_alternative(
        baseline_on_foot_miles=7.9,
        baseline_elapsed_p75_minutes=128,
        alternative_on_foot_miles=6.7,
        alternative_elapsed_p75_minutes=140,
        component_on_foot_miles=[4.7, 2.0],
        parking_blockers=["street parking requires manual check"],
        car_access_benefit="refill_bail",
    )

    assert classification["status"] == "needs_parking_check"
    assert classification["underlying_status"] == "promising"
    assert classification["on_foot_savings_miles"] == 1.2
    assert classification["elapsed_delta_minutes"] == 12


def test_classify_alternative_does_not_keep_short_component_only_splits():
    classification = audit.classify_alternative(
        baseline_on_foot_miles=5.0,
        baseline_elapsed_p75_minutes=100,
        alternative_on_foot_miles=4.9,
        alternative_elapsed_p75_minutes=108,
        component_on_foot_miles=[3.0, 1.9],
        parking_blockers=[],
        car_access_benefit="refill_bail",
    )

    assert classification["status"] == "not_worth_it"
    assert classification["underlying_status"] == "not_worth_it"


def test_classify_alternative_rejects_small_generic_refill_bail_savings():
    classification = audit.classify_alternative(
        baseline_on_foot_miles=5.0,
        baseline_elapsed_p75_minutes=100,
        alternative_on_foot_miles=4.6,
        alternative_elapsed_p75_minutes=106,
        component_on_foot_miles=[2.3, 2.3],
        parking_blockers=[],
        car_access_benefit="refill_bail",
    )

    assert classification["status"] == "not_worth_it"
    assert classification["underlying_status"] == "not_worth_it"


def test_classify_alternative_keeps_substantial_savings_with_reasonable_car_penalty():
    classification = audit.classify_alternative(
        baseline_on_foot_miles=7.93,
        baseline_elapsed_p75_minutes=128,
        alternative_on_foot_miles=5.47,
        alternative_elapsed_p75_minutes=160,
        component_on_foot_miles=[1.36, 4.11],
        parking_blockers=["parking inferred from trailhead layer"],
        car_access_benefit="refill_bail",
    )

    assert classification["status"] == "needs_parking_check"
    assert classification["underlying_status"] == "promising"
    assert classification["on_foot_savings_miles"] == 2.46
    assert classification["elapsed_delta_minutes"] == 32


def test_label_for_component_derives_lettered_package_labels():
    package = {"package_number": 1, "components": [{}, {}]}

    assert audit.label_for_component(package, {}, 0) == "1A"
    assert audit.label_for_component(package, {}, 1) == "1B"


def test_trail_variant_sequences_reverse_order_keeps_nonreversible_ascent_trails():
    reversible = {
        "trail_name": "Reversible",
        "segments": [
            {
                "seg_id": 1,
                "direction": "both",
                "start": (0.0, 0.0),
                "end": (0.1, 0.1),
                "coordinates": [(0.0, 0.0), (0.1, 0.1)],
            }
        ],
        "remaining_segment_ids": [1],
        "start": (0.0, 0.0),
        "end": (0.1, 0.1),
    }
    ascent = {
        "trail_name": "Ascent Only",
        "segments": [
            {
                "seg_id": 2,
                "direction": "ascent",
                "start": (0.2, 0.2),
                "end": (0.3, 0.3),
                "coordinates": [(0.2, 0.2), (0.3, 0.3)],
            }
        ],
        "remaining_segment_ids": [2],
        "start": (0.2, 0.2),
        "end": (0.3, 0.3),
    }
    trails = [reversible, ascent, reversible | {"trail_name": "Reversible 2"}, ascent | {"trail_name": "Ascent Only 2"}]

    variants = audit.trail_variant_sequences(trails)

    assert [trail["trail_name"] for trail in variants[1]] == [
        "Ascent Only 2",
        "Reversible 2",
        "Ascent Only",
        "Reversible",
    ]
    assert {segment["seg_id"] for trail in variants[1] for segment in trail["segments"]} == {1, 2}


def test_build_alternative_rejects_same_anchor_splits():
    anchor = {
        "anchor_id": "same",
        "name": "Same Lot",
        "lat": 43.1,
        "lon": -116.1,
        "parking_confidence": "source_validated_trailhead",
        "field_ready": True,
    }
    baseline = {"on_foot_miles": 8.0, "elapsed_p75_minutes": 150}
    component = {
        "trail_names": ["Short Trail"],
        "start_anchor": anchor,
        "official_miles": 1.0,
        "on_foot_miles": 2.0,
        "p75_activity_minutes_without_home_drive": 25,
        "parking_minutes": 8,
        "drive_to_trailhead_minutes": 5,
        "return_drive_minutes": 5,
        "parking_blockers": [],
    }

    alternative = audit.build_alternative(
        outing_label="1A",
        partition_index=1,
        baseline=baseline,
        left=component,
        right=component,
        drive_model={"straight_line_factor": 1.2, "minutes_per_mile": 2.0, "minimum_one_way_minutes": 4},
    )

    assert alternative["status"] == "not_worth_it"
    assert alternative["recommendation"] == "Rejected: both components use the same parked start."


def test_public_summary_removes_private_coordinates_and_raw_activity_ids():
    report = {
        "alternatives": [
            {
                "alternative_id": "1A-MS-01",
                "components": [
                    {
                        "start_anchor": {
                            "anchor_id": "strava-parking-anchor-01",
                            "name": "Strava parking anchor 01",
                            "lat": 43.1,
                            "lon": -116.1,
                            "privacy": "private_exact_coordinates",
                            "raw_activity_ids": [123],
                        }
                    }
                ],
            }
        ]
    }

    public = audit.public_safe_report(report)
    payload = repr(public)

    assert "43.1" not in payload
    assert "-116.1" not in payload
    assert "raw_activity_ids" not in payload
    anchor = public["alternatives"][0]["components"][0]["start_anchor"]
    assert anchor["name"] == "Private Strava parking anchor"
    assert anchor["privacy"] == "private_exact_coordinates"
