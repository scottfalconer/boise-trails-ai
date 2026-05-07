import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "field_route_walkthrough_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("field_route_walkthrough_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def edge(module, edge_id, name, coords, *, source_class="r2r_trail", segment_id=None, direction="both", props=None):
    return module.TrailEdge(
        edge_id=edge_id,
        name=name,
        normalized_name=module.normalize_name(name),
        signposts=module.extract_signposts(name),
        source_class=source_class,
        coords=coords,
        segment_id=str(segment_id) if segment_id is not None else None,
        direction=direction,
        raw_properties=props or {},
    )


def route(cue_text, *, segment_ids=("51",)):
    return {
        "outing_id": "test-route",
        "label": "Test",
        "trailhead": "Test Trailhead",
        "segment_ids": list(segment_ids),
        "parking": {"lon": 0.0, "lat": 0.0, "name": "Test Trailhead"},
        "turn_by_turn_steps": [{"kind": "access", "title": cue_text}],
        "wayfinding_cues": [],
    }


def test_start_access_named_edge_required():
    module = load_module()
    access = edge(module, "access", "#57 Harrison Hollow", [(0.0, 0.0), (0.001, 0.0)])
    official = edge(
        module,
        "official-51",
        "#51 Who Now Loop",
        [(0.001, 0.0), (0.002, 0.0)],
        source_class="official_segment",
        segment_id="51",
    )

    result = module.audit_route_walkthrough(
        route("Leave car toward #51 Who Now Loop Trail."),
        [[(0.0, 0.0), (0.001, 0.0), (0.002, 0.0)]],
        [access, official],
        {"51": official},
        snap_tolerance_miles=0.002,
    )

    assert result["passed"] is False
    assert {failure["code"] for failure in result["failures"]} == {"start_access_missing_named_edge"}
    assert "#57 Harrison Hollow" in result["failures"][0]["expected_cue_text_hint"]


def test_start_access_named_edge_passes_when_cued():
    module = load_module()
    access = edge(module, "access", "#57 Harrison Hollow", [(0.0, 0.0), (0.001, 0.0)])
    official = edge(
        module,
        "official-51",
        "#51 Who Now Loop",
        [(0.001, 0.0), (0.002, 0.0)],
        source_class="official_segment",
        segment_id="51",
    )

    result = module.audit_route_walkthrough(
        route("Take #57 Harrison Hollow until the signed #51 Who Now Loop junction."),
        [[(0.0, 0.0), (0.001, 0.0), (0.002, 0.0)]],
        [access, official],
        {"51": official},
        snap_tolerance_miles=0.002,
    )

    assert result["passed"] is True


def test_between_connector_named_edge_required():
    module = load_module()
    official_a = edge(
        module,
        "official-a",
        "Trail A",
        [(0.0, 0.0), (0.001, 0.0)],
        source_class="official_segment",
        segment_id="1",
    )
    connector = edge(module, "connector", "Connector Y", [(0.001, 0.0), (0.002, 0.0)])
    official_b = edge(
        module,
        "official-b",
        "Trail Z",
        [(0.002, 0.0), (0.003, 0.0)],
        source_class="official_segment",
        segment_id="2",
    )

    result = module.audit_route_walkthrough(
        route("Follow Trail A, then take Trail Z.", segment_ids=("1", "2")),
        [[(0.0, 0.0), (0.001, 0.0), (0.002, 0.0), (0.003, 0.0)]],
        [official_a, connector, official_b],
        {"1": official_a, "2": official_b},
        snap_tolerance_miles=0.002,
    )

    assert result["passed"] is False
    assert "named_connector_not_cued" in {failure["code"] for failure in result["failures"]}


def test_generic_osm_connector_id_is_not_treated_as_field_signage():
    module = load_module()
    official_a = edge(
        module,
        "official-a",
        "Trail A",
        [(0.0, 0.0), (0.001, 0.0)],
        source_class="official_segment",
        segment_id="1",
    )
    connector = edge(
        module,
        "connector",
        "OSM footway connector 72484",
        [(0.001, 0.0), (0.002, 0.0)],
        source_class="osm_path_footway",
    )
    official_b = edge(
        module,
        "official-b",
        "Trail Z",
        [(0.002, 0.0), (0.003, 0.0)],
        source_class="official_segment",
        segment_id="2",
    )

    result = module.audit_route_walkthrough(
        route("Follow Trail A, then take Trail Z.", segment_ids=("1", "2")),
        [[(0.0, 0.0), (0.001, 0.0), (0.002, 0.0), (0.003, 0.0)]],
        [official_a, connector, official_b],
        {"1": official_a, "2": official_b},
        snap_tolerance_miles=0.002,
    )

    assert result["passed"] is True


def test_hidden_track_gap_fails_even_if_each_trackseg_has_small_internal_gaps():
    module = load_module()
    official_a = edge(
        module,
        "official-a",
        "Trail A",
        [(0.0, 0.0), (0.001, 0.0)],
        source_class="official_segment",
        segment_id="1",
    )
    official_b = edge(
        module,
        "official-b",
        "Trail B",
        [(0.02, 0.0), (0.021, 0.0)],
        source_class="official_segment",
        segment_id="2",
    )

    result = module.audit_route_walkthrough(
        route("Follow Trail A, then Trail B.", segment_ids=("1", "2")),
        [[(0.0, 0.0), (0.001, 0.0)], [(0.02, 0.0), (0.021, 0.0)]],
        [official_a, official_b],
        {"1": official_a, "2": official_b},
        snap_tolerance_miles=0.002,
        max_gap_miles=0.05,
    )

    assert result["passed"] is False
    assert "hidden_track_gap" in {failure["code"] for failure in result["failures"]}


def test_official_multilinestring_parts_are_not_fake_connected():
    module = load_module()
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"segId": "1", "segName": "Split Trail", "direction": "both"},
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [
                        [[0.0, 0.0], [0.001, 0.0]],
                        [[0.01, 0.0], [0.011, 0.0]],
                    ],
                },
            }
        ],
    }

    edges, official_index = module.edges_from_official_geojson(geojson)

    assert len(edges) == 2
    assert all(edge.coords != [(0.001, 0.0), (0.01, 0.0)] for edge in edges)
    assert len(official_index["1"].parts) == 2


def test_claimed_segment_geometry_coverage_required():
    module = load_module()
    official = edge(
        module,
        "official-1",
        "Official Trail",
        [(0.0, 0.0), (0.004, 0.0)],
        source_class="official_segment",
        segment_id="1",
    )

    result = module.audit_route_walkthrough(
        route("Follow Official Trail.", segment_ids=("1",)),
        [[(0.0, 0.0), (0.002, 0.0)]],
        [official],
        {"1": official},
        snap_tolerance_miles=0.002,
    )

    assert result["passed"] is False
    assert "claimed_segment_not_covered" in {failure["code"] for failure in result["failures"]}


def test_direction_rule_violation():
    module = load_module()
    official = edge(
        module,
        "official-1",
        "Ascent Trail",
        [(0.0, 0.0), (0.002, 0.0)],
        source_class="official_segment",
        segment_id="1",
        direction="ascent",
    )

    result = module.audit_route_walkthrough(
        route("Follow Ascent Trail.", segment_ids=("1",)),
        [[(0.002, 0.0), (0.0, 0.0)]],
        [official],
        {"1": official},
        snap_tolerance_miles=0.002,
    )

    assert result["passed"] is False
    assert "direction_rule_violated" in {failure["code"] for failure in result["failures"]}


def test_direction_rule_allows_reverse_when_route_evidence_says_ascent_is_opposite_official_geometry():
    module = load_module()
    official = edge(
        module,
        "official-1",
        "Ascent Trail",
        [(0.0, 0.0), (0.002, 0.0)],
        source_class="official_segment",
        segment_id="1",
        direction="ascent",
    )
    audited_route = route("Follow Ascent Trail uphill.", segment_ids=("1",))
    audited_route["segment_direction_evidence"] = {
        "1": {
            "direction_rule": "ascent",
            "allowed_geometry_direction": "reverse",
            "direction_cue": "ASCENT REQUIRED: follow map arrows opposite official geometry.",
        }
    }

    result = module.audit_route_walkthrough(
        audited_route,
        [[(0.002, 0.0), (0.0, 0.0)]],
        [official],
        {"1": official},
        snap_tolerance_miles=0.002,
    )

    assert result["passed"] is True


def test_direction_rule_uses_matched_official_pass_not_nearest_repeated_endpoint():
    module = load_module()
    access = edge(
        module,
        "access",
        "Access Trail",
        [(0.002, 0.0), (0.0, 0.0)],
        source_class="r2r_trail",
    )
    official = edge(
        module,
        "official-1",
        "Ascent Trail",
        [(0.0, 0.0), (0.002, 0.0)],
        source_class="official_segment",
        segment_id="1",
        direction="ascent",
    )

    result = module.audit_route_walkthrough(
        route("Follow Access Trail to Ascent Trail, then climb Ascent Trail.", segment_ids=("1",)),
        [[(0.002, 0.0), (0.0, 0.0), (0.002, 0.0)]],
        [access, official],
        {"1": official},
        snap_tolerance_miles=0.002,
    )

    assert result["passed"] is True


def test_blocked_connector_used():
    module = load_module()
    blocked = edge(
        module,
        "blocked",
        "Private Connector",
        [(0.0, 0.0), (0.001, 0.0)],
        source_class="osm_path_footway",
        props={"access": "no"},
    )
    official = edge(
        module,
        "official-1",
        "Official Trail",
        [(0.001, 0.0), (0.002, 0.0)],
        source_class="official_segment",
        segment_id="1",
    )

    result = module.audit_route_walkthrough(
        route("Follow Private Connector to Official Trail.", segment_ids=("1",)),
        [[(0.0, 0.0), (0.001, 0.0), (0.002, 0.0)]],
        [blocked, official],
        {"1": official},
        snap_tolerance_miles=0.002,
    )

    assert result["passed"] is False
    assert "blocked_connector_used" in {failure["code"] for failure in result["failures"]}
