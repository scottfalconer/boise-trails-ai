import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "post_credit_connector_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("post_credit_connector_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_gpx(path, coords):
    path.parent.mkdir(parents=True, exist_ok=True)
    points = "\n".join(f'<trkpt lat="{lat}" lon="{lon}"></trkpt>' for lon, lat in coords)
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
  <trk><trkseg>
{points}
  </trkseg></trk>
</gpx>
""",
        encoding="utf-8",
    )


def graph_with_edges(edges):
    graph = {}
    nodes = set()
    for left, right, distance in edges:
        nodes.add(left)
        nodes.add(right)
        graph.setdefault(left, []).append(
            {
                "to": right,
                "distance": distance,
                "name": "Connector",
                "edge_type": "connector",
                "connector_class": "test_connector",
                "source": "test",
            }
        )
        graph.setdefault(right, []).append(
            {
                "to": left,
                "distance": distance,
                "name": "Connector",
                "edge_type": "connector",
                "connector_class": "test_connector",
                "source": "test",
            }
        )
    return {"graph": graph, "nodes": sorted(nodes)}


def graph_with_custom_edges(edges):
    graph = {}
    nodes = set()
    for edge in edges:
        left = edge["from"]
        right = edge["to"]
        nodes.add(left)
        nodes.add(right)
        graph.setdefault(left, []).append(
            {
                "to": right,
                "distance": edge["distance"],
                "name": edge.get("name", "Connector"),
                "edge_type": edge.get("edge_type", "connector"),
                "connector_class": edge.get("connector_class", "test_connector"),
                "source": "test",
                "seg_id": edge.get("seg_id"),
            }
        )
        if edge.get("bidirectional", True):
            graph.setdefault(right, []).append(
                {
                    "to": left,
                    "distance": edge["distance"],
                    "name": edge.get("name", "Connector"),
                    "edge_type": edge.get("edge_type", "connector"),
                    "connector_class": edge.get("connector_class", "test_connector"),
                    "source": "test",
                    "seg_id": edge.get("seg_id"),
                }
            )
    return {"graph": graph, "nodes": sorted(nodes)}


def build_audit(tmp_path, route, connector_graph):
    module = load_module()
    return module.build_post_credit_connector_audit(
        {"routes": [route]},
        packet_dir=tmp_path / "packet",
        connector_graph=connector_graph,
        distance_tolerance_miles=0.005,
        snap_tolerance_miles=0.01,
    )


def line_miles(coords):
    return load_module().line_length_miles(coords)


def test_official_credit_cue_with_extra_movement_fails(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    end = (-115.98, 43.0)
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid, end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": 0.4,
                "source_leg_miles": 0.4,
                "source_path_coordinates": [start, mid, end],
                "official_miles": 0.2,
            }
        ],
    }

    audit = build_audit(tmp_path, route, graph_with_edges([]))

    assert audit["status"] == "passed"
    assert audit["summary"]["hidden_exit_finding_count"] == 0
    assert audit["summary"]["hidden_exit_warning_count"] == 1
    warning = audit["warnings"][0]
    assert warning["code"] == "official_credit_cue_hides_post_credit_exit"
    assert warning["hidden_exit_feet"] == 1056


def test_official_credit_display_mileage_without_source_path_is_not_hidden_exit_evidence(tmp_path):
    packet_dir = tmp_path / "packet"
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [(-116.0, 43.0), (-115.99, 43.0)])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": 0.6,
                "source_leg_miles": 0.6,
                "official_miles": 0.2,
            }
        ],
    }

    audit = build_audit(tmp_path, route, graph_with_edges([]))

    assert audit["status"] == "passed"
    assert audit["summary"]["hidden_exit_finding_count"] == 0
    assert audit["summary"]["hidden_exit_warning_count"] == 0


def test_hidden_exit_uses_source_miles_not_scaled_display_miles(tmp_path):
    packet_dir = tmp_path / "packet"
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [(-116.0, 43.0), (-115.99, 43.0)])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "official_miles": 0.4,
                "source_leg_miles": 0.4,
                "leg_miles": 0.5,
                "route_miles": 0.0,
                "route_leg_miles": 0.4,
            }
        ],
    }

    audit = build_audit(tmp_path, route, graph_with_edges([]))

    assert audit["status"] == "passed"
    assert audit["summary"]["hidden_exit_finding_count"] == 0
    assert audit["summary"]["hidden_exit_warning_count"] == 0


def test_explicit_post_credit_connector_passes_when_shortest(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    end = (-115.98, 43.0)
    first_leg = line_miles([start, mid])
    connector_leg = line_miles([mid, end])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid, end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101, 102],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
                "source_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "connector_named_trail",
                "route_miles": first_leg,
                "route_leg_miles": connector_leg,
            },
        ],
    }
    connector_graph = graph_with_edges([(mid, end, connector_leg)])

    audit = build_audit(tmp_path, route, connector_graph)

    assert audit["status"] == "passed"
    assert audit["summary"]["post_credit_connector_proof_count"] == 1
    assert audit["routes"][0]["post_credit_connector_proofs"][0]["status"] == "passed"


def test_explicit_post_credit_connector_fails_when_packet_savings_metadata_is_stale(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    end = (-115.98, 43.0)
    first_leg = line_miles([start, mid])
    connector_leg = line_miles([mid, end])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid, end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101, 102],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
                "source_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "connector_named_trail",
                "route_miles": first_leg,
                "route_leg_miles": connector_leg,
                "shortest_repair_savings_miles": 0.25,
            },
        ],
    }
    connector_graph = graph_with_edges([(mid, end, connector_leg)])

    audit = build_audit(tmp_path, route, connector_graph)

    assert audit["status"] == "failed"
    assert audit["summary"]["stale_connector_savings_finding_count"] == 1
    finding = audit["findings"][0]
    assert finding["failure_code"] == "field_packet_connector_savings_mismatch"
    assert finding["field_packet_shortest_repair_savings_miles"] == 0.25
    assert finding["savings_miles"] == 0.0


def test_explicit_post_credit_connector_uses_gpx_route_interval_not_scaled_source_miles(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    end = (-115.98, 43.0)
    first_leg = line_miles([start, mid])
    connector_leg = line_miles([mid, end])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid, end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101, 102],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
                "source_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "connector_named_trail",
                "route_miles": first_leg,
                "route_leg_miles": connector_leg,
                "source_cum_miles": first_leg,
                "source_leg_miles": connector_leg * 2,
            },
        ],
    }
    connector_graph = graph_with_edges([(mid, end, connector_leg)])

    audit = build_audit(tmp_path, route, connector_graph)

    assert audit["status"] == "passed"
    proof = audit["routes"][0]["post_credit_connector_proofs"][0]
    assert proof["status"] == "passed"
    assert proof["actual_miles"] == round(connector_leg, 4)


def test_explicit_post_credit_connector_uses_route_interval_before_display_leg(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    end = (-115.98, 43.0)
    first_leg = line_miles([start, mid])
    connector_leg = line_miles([mid, end])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid, end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101, 102],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "connector_named_trail",
                "route_miles": first_leg,
                "route_leg_miles": connector_leg,
                "leg_miles": connector_leg * 4,
            },
        ],
    }
    connector_graph = graph_with_edges([(mid, end, connector_leg)])

    audit = build_audit(tmp_path, route, connector_graph)

    assert audit["status"] == "passed"
    proof = audit["routes"][0]["post_credit_connector_proofs"][0]
    assert proof["actual_miles"] == round(connector_leg, 4)


def test_explicit_post_credit_connector_uses_source_path_geometry_when_present(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    source_end = (-115.98, 43.0)
    gpx_detour = (-115.985, 43.01)
    first_leg = line_miles([start, mid])
    source_leg = line_miles([mid, source_end])
    inflated_route_leg = line_miles([mid, gpx_detour, source_end])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid, gpx_detour, source_end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101, 102],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "connector_named_trail",
                "route_miles": first_leg,
                "route_leg_miles": inflated_route_leg,
                "leg_miles": source_leg,
                "source_path_coordinates": [mid, source_end],
            },
        ],
    }
    connector_graph = graph_with_edges([(mid, source_end, source_leg)])

    audit = build_audit(tmp_path, route, connector_graph)

    assert audit["status"] == "passed"
    proof = audit["routes"][0]["post_credit_connector_proofs"][0]
    assert proof["actual_miles"] == round(source_leg, 4)


def test_tiny_source_path_without_graph_match_passes_inside_snap_tolerance(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    tiny_end = (-115.9899, 43.0)
    first_leg = line_miles([start, mid])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid, tiny_end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "return_to_car",
                "source_leg_miles": 0.01,
                "source_path_coordinates": [mid, tiny_end],
            },
        ],
    }

    audit = build_audit(tmp_path, route, graph_with_edges([]))

    assert audit["status"] == "passed"
    assert audit["summary"]["unproved_connector_finding_count"] == 0


def test_zero_length_return_ignores_scaled_display_delta(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    first_leg = line_miles([start, mid])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "return_to_car",
                "leg_miles": 0.5,
                "source_leg_miles": 0,
                "route_miles": first_leg,
                "route_leg_miles": 0,
            },
        ],
    }

    audit = build_audit(tmp_path, route, graph_with_edges([]))

    assert audit["status"] == "passed"
    assert audit["summary"]["post_credit_connector_proof_count"] == 0


def test_zero_length_exit_ignores_scaled_display_delta(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    first_leg = line_miles([start, mid])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "exit_access",
                "leg_miles": 0.5,
                "source_leg_miles": 0,
                "route_miles": first_leg,
                "route_leg_miles": 0,
            },
        ],
    }

    audit = build_audit(tmp_path, route, graph_with_edges([]))

    assert audit["status"] == "passed"
    assert audit["summary"]["post_credit_connector_proof_count"] == 0


def test_credited_official_repeat_return_passes_when_directional_graph_cannot_reverse(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    end = (-115.99, 43.0)
    leg = line_miles([start, end])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, end, start])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": leg,
            },
            {
                "seq": 2,
                "cue_type": "exit_access",
                "route_miles": leg,
                "route_leg_miles": leg,
                "official_repeat_miles": leg,
                "official_repeat_segment_ids": [101],
                "source_path_coordinates": [end, start],
            },
        ],
    }
    connector_graph = graph_with_custom_edges(
        [
            {
                "from": start,
                "to": end,
                "distance": leg,
                "edge_type": "official_repeat",
                "connector_class": "official_repeat",
                "seg_id": 101,
                "bidirectional": False,
            }
        ]
    )

    audit = build_audit(tmp_path, route, connector_graph)

    assert audit["status"] == "passed"
    proof = audit["routes"][0]["post_credit_connector_proofs"][0]
    assert proof["credited_official_repeat_return_proven"] is True


def test_target_official_sliver_inside_snap_tolerance_does_not_block_connector_proof(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    official_end = (-115.99, 43.0)
    connector_end = (-115.9899, 43.0)
    first_leg = line_miles([(-116.01, 43.0), start])
    connector_leg = line_miles([start, official_end, connector_end])
    sliver_leg = line_miles([official_end, connector_end])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [(-116.01, 43.0), start, official_end, connector_end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101, 102],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "connector_named_trail",
                "route_miles": first_leg,
                "route_leg_miles": connector_leg,
            },
        ],
    }
    connector_graph = graph_with_custom_edges(
        [
            {"from": start, "to": official_end, "distance": connector_leg - sliver_leg},
            {
                "from": official_end,
                "to": connector_end,
                "distance": sliver_leg,
                "edge_type": "official_repeat",
                "connector_class": "official_repeat",
                "seg_id": 102,
            },
        ]
    )

    audit = build_audit(tmp_path, route, connector_graph)

    assert audit["status"] == "passed"
    proof = audit["routes"][0]["post_credit_connector_proofs"][0]
    assert proof["target_official_sliver_within_snap_tolerance"] is True


def test_explicit_source_connector_passes_when_graph_has_no_alternative(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    end = (-115.98, 43.0)
    first_leg = line_miles([start, mid])
    connector_leg = line_miles([mid, end])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid, end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101, 102],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "connector_named_trail",
                "target": "Mapped Connector",
                "signed_as": ["Mapped Connector"],
                "source_path_coordinates": [mid, end],
                "source_leg_miles": connector_leg,
                "route_miles": first_leg,
                "route_leg_miles": connector_leg,
            },
        ],
    }

    audit = build_audit(tmp_path, route, graph_with_edges([]))

    assert audit["status"] == "passed"
    proof = audit["routes"][0]["post_credit_connector_proofs"][0]
    assert proof["source_geometry_without_graph_alternative"] is True


def test_explicit_post_credit_connector_fails_when_shorter_legal_path_exists(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    detour = (-115.985, 43.005)
    end = (-115.98, 43.0)
    first_leg = line_miles([start, mid])
    connector_leg = line_miles([mid, detour, end])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid, detour, end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101, 102],
        "gpx_href": "gpx/official/route-a.gpx",
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
                "source_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "connector_named_trail",
                "route_miles": first_leg,
                "route_leg_miles": connector_leg,
            },
        ],
    }
    connector_graph = graph_with_edges([(mid, end, 0.2), (mid, detour, 0.4), (detour, end, 0.4)])

    audit = build_audit(tmp_path, route, connector_graph)

    assert audit["status"] == "failed"
    assert audit["summary"]["shorter_connector_finding_count"] == 1
    finding = audit["findings"][0]
    assert finding["failure_code"] == "shorter_legal_connector_found"
    assert finding["savings_feet"] > 0


def test_source_gap_repaired_route_fails_once_without_misleading_connector_savings(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    end = (-115.98, 43.0)
    first_leg = line_miles([start, mid])
    connector_leg = line_miles([mid, end])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid, end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101, 102],
        "gpx_href": "gpx/official/route-a.gpx",
        "source_gap_repair": {
            "raw_inter_segment_gap_count": 1,
            "repaired_inter_segment_gap_count": 1,
            "remaining_inter_segment_gap_count": 0,
            "repair_method": "graph_connector_stitch",
        },
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
                "source_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "connector_named_trail",
                "route_miles": first_leg,
                "route_leg_miles": connector_leg,
            },
        ],
    }
    connector_graph = graph_with_edges([(mid, end, connector_leg / 2)])

    audit = build_audit(tmp_path, route, connector_graph)

    assert audit["status"] == "failed"
    assert audit["summary"]["source_gap_proof_blocker_count"] == 1
    assert audit["summary"]["shorter_connector_finding_count"] == 0
    assert audit["summary"]["post_credit_connector_proof_count"] == 0
    assert audit["findings"][0]["code"] == "route_source_gap_repair_prevents_post_credit_proof"


def test_route_card_gpx_mismatch_warns_without_suppressing_connector_proof(tmp_path):
    packet_dir = tmp_path / "packet"
    start = (-116.0, 43.0)
    mid = (-115.99, 43.0)
    detour = (-115.985, 43.005)
    end = (-115.98, 43.0)
    first_leg = line_miles([start, mid])
    connector_leg = line_miles([mid, detour, end])
    write_gpx(packet_dir / "gpx" / "official" / "route-a.gpx", [start, mid, detour, end])
    route = {
        "outing_id": "route-a",
        "label": "Route A",
        "segment_ids": [101, 102],
        "gpx_href": "gpx/official/route-a.gpx",
        "on_foot_miles": first_leg + 0.1,
        "wayfinding_cues": [
            {
                "seq": 1,
                "cue_type": "follow_official_segment",
                "official_segment_ids": [101],
                "route_miles": 0.0,
                "route_leg_miles": first_leg,
                "source_leg_miles": first_leg,
            },
            {
                "seq": 2,
                "cue_type": "connector_named_trail",
                "route_miles": first_leg,
                "route_leg_miles": connector_leg,
            },
        ],
    }
    connector_graph = graph_with_edges([(mid, end, 0.2), (mid, detour, 0.4), (detour, end, 0.4)])

    audit = build_audit(tmp_path, route, connector_graph)

    assert audit["status"] == "failed"
    assert audit["summary"]["route_card_gpx_mismatch_count"] == 1
    assert audit["summary"]["warning_count"] == 1
    assert audit["summary"]["shorter_connector_finding_count"] == 1
    assert audit["summary"]["post_credit_connector_proof_count"] == 1
    assert audit["warnings"][0]["code"] == "route_card_gpx_mileage_mismatch_warning"
    assert audit["findings"][0]["failure_code"] == "shorter_legal_connector_found"
