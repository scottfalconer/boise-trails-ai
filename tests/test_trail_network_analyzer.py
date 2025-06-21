import pytest

from trail_route_ai.planner_utils import Edge
from trail_route_ai.trail_network_analyzer import identify_natural_trail_groups, find_loops

# Dummy coordinates and other values for Edge instances
coords1 = [(0,0), (1,1)]
coords2 = [(1,1), (2,2)]
coords3 = [(2,2), (3,3)]
coords4 = [(3,3), (4,4)]
coords5 = [(4,4), (5,5)]
coords6 = [(5,5), (6,6)]
coords7 = [(6,6), (7,7)]

@pytest.fixture
def sample_segments():
    return [
        Edge(seg_id="1", name="Dry Creek Trail No. 1", start=(0,0), end=(1,1), length_mi=1, elev_gain_ft=100, coords=coords1),
        Edge(seg_id="2", name="Dry Creek Trail #2", start=(1,1), end=(2,2), length_mi=1, elev_gain_ft=100, coords=coords2),
        Edge(seg_id="3", name="Dry Creek", start=(2,2), end=(3,3), length_mi=1, elev_gain_ft=100, coords=coords3), # Should group with Dry Creek
        Edge(seg_id="4", name="Shingle Creek Trail 1", start=(3,3), end=(4,4), length_mi=1, elev_gain_ft=100, coords=coords4),
        Edge(seg_id="5", name="Shingle Creek Path", start=(4,4), end=(5,5), length_mi=1, elev_gain_ft=100, coords=coords5), # Should group with Shingle Creek
        Edge(seg_id="6", name="Another Trail", start=(5,5), end=(6,6), length_mi=1, elev_gain_ft=100, coords=coords6),
        Edge(seg_id="7", name="Trail 10", start=(6,6), end=(7,7), length_mi=1, elev_gain_ft=100, coords=coords7), # Should group as "Trail"
        Edge(seg_id="8", name=None, start=(0,0), end=(1,1), length_mi=1, elev_gain_ft=100, coords=coords1),
        Edge(seg_id="9", name="Table Rock Loop", start=(1,1), end=(2,2), length_mi=1, elev_gain_ft=100, coords=coords2), # Should group as "Table Rock"
        Edge(seg_id="10", name="Camel's Back Trail", start=(2,2), end=(3,3), length_mi=1, elev_gain_ft=100, coords=coords3), # Should group as "Camel's Back"
        Edge(seg_id="11", name="Five Mile Gulch Spur", start=(3,3), end=(4,4), length_mi=1, elev_gain_ft=100, coords=coords4), # Should group as "Five Mile Gulch"
    ]

def test_identify_natural_trail_groups(sample_segments):
    grouped_trails = identify_natural_trail_groups(sample_segments)

    assert "Dry Creek" in grouped_trails
    assert len(grouped_trails["Dry Creek"]) == 3
    assert sample_segments[0] in grouped_trails["Dry Creek"]
    assert sample_segments[1] in grouped_trails["Dry Creek"]
    assert sample_segments[2] in grouped_trails["Dry Creek"]

    assert "Shingle Creek" in grouped_trails
    assert len(grouped_trails["Shingle Creek"]) == 2
    assert sample_segments[3] in grouped_trails["Shingle Creek"]
    assert sample_segments[4] in grouped_trails["Shingle Creek"]

    assert "Another" in grouped_trails
    assert len(grouped_trails["Another"]) == 1
    assert sample_segments[5] in grouped_trails["Another"]

    assert "Trail" in grouped_trails
    assert len(grouped_trails["Trail"]) == 1
    assert sample_segments[6] in grouped_trails["Trail"]

    assert "Unnamed Trails" in grouped_trails
    assert len(grouped_trails["Unnamed Trails"]) == 1
    assert sample_segments[7] in grouped_trails["Unnamed Trails"]

    assert "Table Rock" in grouped_trails
    assert len(grouped_trails["Table Rock"]) == 1
    assert sample_segments[8] in grouped_trails["Table Rock"]

    assert "Camel's Back" in grouped_trails
    assert len(grouped_trails["Camel's Back"]) == 1
    assert sample_segments[9] in grouped_trails["Camel's Back"]

    assert "Five Mile Gulch" in grouped_trails
    assert len(grouped_trails["Five Mile Gulch"]) == 1
    assert sample_segments[10] in grouped_trails["Five Mile Gulch"]

    assert len(grouped_trails) == 8

def test_identify_natural_trail_groups_empty():
    grouped_trails = identify_natural_trail_groups([])
    assert grouped_trails == {}

def test_identify_natural_trail_groups_no_names():
    segments = [
        Edge(seg_id="1", name=None, start=(0,0), end=(1,1), length_mi=1, elev_gain_ft=100, coords=coords1),
        Edge(seg_id="2", name=None, start=(1,1), end=(2,2), length_mi=1, elev_gain_ft=100, coords=coords2),
    ]
    grouped_trails = identify_natural_trail_groups(segments)
    assert "Unnamed Trails" in grouped_trails
    assert len(grouped_trails["Unnamed Trails"]) == 2

def test_identify_natural_trail_groups_complex_names():
    segments = [
        Edge(seg_id="1", name="Main Street Trail Bypass No. 15", start=(0,0), end=(1,1), length_mi=1, elev_gain_ft=100, coords=coords1),
        Edge(seg_id="2", name="Main Street Trail", start=(1,1), end=(2,2), length_mi=1, elev_gain_ft=100, coords=coords2),
        Edge(seg_id="3", name="Mountain View Path Section 2", start=(2,2), end=(3,3), length_mi=1, elev_gain_ft=100, coords=coords3),
        Edge(seg_id="4", name="Mountain View Connector", start=(3,3), end=(4,4), length_mi=1, elev_gain_ft=100, coords=coords4),
    ]
    grouped_trails = identify_natural_trail_groups(segments)

    assert "Main Street" in grouped_trails
    assert len(grouped_trails["Main Street"]) == 1
    assert segments[1] in grouped_trails["Main Street"]

    assert "Main Street Trail Bypass" in grouped_trails
    assert len(grouped_trails["Main Street Trail Bypass"]) == 1
    assert segments[0] in grouped_trails["Main Street Trail Bypass"]

    assert "Mountain View" in grouped_trails
    assert len(grouped_trails["Mountain View"]) == 2
    assert segments[2] in grouped_trails["Mountain View"]
    assert segments[3] in grouped_trails["Mountain View"]

    assert len(grouped_trails) == 3

# --- Tests for find_loops ---
import networkx as nx # type: ignore

# Node definitions (lon, lat)
A = (0.0, 0.0)
B = (1.0, 0.0)
C = (1.0, 1.0)
D = (2.0, 1.0)
E = (3.0, 1.0)
F = (3.0, 0.0)
G_node = (0.0, 1.0) # Another node for a more complex graph

@pytest.fixture
def sample_graph_and_segments_for_loops():
    s1 = Edge(seg_id="s1", name="TrailA-B", start=A, end=B, length_mi=1, elev_gain_ft=10, coords=[A,B])
    s2 = Edge(seg_id="s2", name="TrailB-C", start=B, end=C, length_mi=1, elev_gain_ft=10, coords=[B,C])
    s3 = Edge(seg_id="s3", name="TrailC-A", start=C, end=A, length_mi=1.4, elev_gain_ft=10, coords=[C,A])

    s4 = Edge(seg_id="s4", name="TrailC-D", start=C, end=D, length_mi=1, elev_gain_ft=10, coords=[C,D])

    s5 = Edge(seg_id="s5", name="TrailD-E", start=D, end=E, length_mi=1, elev_gain_ft=10, coords=[D,E])
    s6 = Edge(seg_id="s6", name="TrailE-F", start=E, end=F, length_mi=1, elev_gain_ft=10, coords=[E,F])
    s7 = Edge(seg_id="s7", name="TrailF-D", start=F, end=D, length_mi=1.4, elev_gain_ft=10, coords=[F,D])

    s8_conn = Edge(seg_id="s8_conn", name="Conn_B-D", start=B, end=D, length_mi=1, elev_gain_ft=5, coords=[B,D], kind="connector")
    s9_conn = Edge(seg_id="s9_conn", name="Conn_F-A", start=F, end=A, length_mi=3, elev_gain_ft=5, coords=[F,A], kind="connector") # longer connector

    # Lasso example segments
    dc1 = Edge(seg_id="dc1", name="Dry Creek 1", start=A, end=B, length_mi=1.0, elev_gain_ft=10, coords=[A,B])
    dc2 = Edge(seg_id="dc2", name="Dry Creek 2", start=B, end=D, length_mi=1.0, elev_gain_ft=10, coords=[B,D]) # B to D (was C')
    sh1 = Edge(seg_id="sh1", name="Shingle 1", start=B, end=C, length_mi=0.7, elev_gain_ft=5, coords=[B,C])   # B to C (was D')
    sh2 = Edge(seg_id="sh2", name="Shingle 2", start=C, end=D, length_mi=0.7, elev_gain_ft=5, coords=[C,D])   # C to D (was D' to C')

    all_segments_for_graph = [s1, s2, s3, s4, s5, s6, s7, s8_conn, s9_conn, dc1, dc2, sh1, sh2]

    graph = nx.DiGraph()
    for seg in all_segments_for_graph:
        # Ensure nodes exist
        graph.add_node(seg.start)
        graph.add_node(seg.end)
        # Add edge with its canonical object
        graph.add_edge(seg.start, seg.end, edge=seg, weight=seg.length_mi)
        if seg.direction == "both": # Assuming all are 'both' for this test setup
            rev_seg = seg.reverse() # Use the reverse method from Edge
            graph.add_edge(seg.end, seg.start, edge=rev_seg, weight=seg.length_mi)

    return {
        "graph": graph,
        "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7,
        "s8_conn": s8_conn, "s9_conn": s9_conn,
        "group_abc": [s1, s2, s3],       # Loop A-B-C-A
        "group_def": [s5, s6, s7],       # Loop D-E-F-D
        "group_bcdb": [s2,s4,s8_conn],   # Loop B-C-D-B via connector s8
        "group_abdfa": [s1,s8_conn,s7,s9_conn], # Loop A-B-D-F-A via connectors
        "lasso_loop_segments": [sh1, sh2, dc2.reverse()], # B-C-D-B (sh1, sh2, reversed dc2)
        "lasso_full_trail_group": [dc1, dc2, sh1, sh2], # For testing if find_loops gets the shingle part
        "dc1": dc1, "dc2": dc2, "sh1": sh1, "sh2": sh2,
    }

def get_edge_ids(loop: list[Edge]) -> set[str | None]:
    return {e.seg_id for e in loop if e.seg_id}

def test_find_loops_simple_loop(sample_graph_and_segments_for_loops):
    graph = sample_graph_and_segments_for_loops["graph"]
    group_abc = sample_graph_and_segments_for_loops["group_abc"]

    loops = find_loops(graph, group_abc, min_segments=3, min_length_mi=3.3, max_length_mi=3.5) # 1+1+1.4 = 3.4

    assert len(loops) == 1
    loop1_ids = get_edge_ids(loops[0])
    assert loop1_ids == {"s1", "s2", "s3"}

def test_find_loops_no_loops_in_group(sample_graph_and_segments_for_loops):
    graph = sample_graph_and_segments_for_loops["graph"]
    non_looping_group = [sample_graph_and_segments_for_loops["s1"], sample_graph_and_segments_for_loops["s4"]]
    loops = find_loops(graph, non_looping_group)
    assert len(loops) == 0

def test_find_loops_multiple_distinct_loops_in_group(sample_graph_and_segments_for_loops):
    graph = sample_graph_and_segments_for_loops["graph"]
    group_with_two_loops = sample_graph_and_segments_for_loops["group_abc"] + sample_graph_and_segments_for_loops["group_def"]

    loops = find_loops(graph, group_with_two_loops, min_segments=3)

    assert len(loops) == 2
    found_loop_abc = any(get_edge_ids(loop) == {"s1", "s2", "s3"} for loop in loops)
    found_loop_def = any(get_edge_ids(loop) == {"s5", "s6", "s7"} for loop in loops)
    assert found_loop_abc
    assert found_loop_def

def test_find_loops_with_connector_forming_loop(sample_graph_and_segments_for_loops):
    graph = sample_graph_and_segments_for_loops["graph"]
    group_bcdb = sample_graph_and_segments_for_loops["group_bcdb"] # s2, s4, s8_conn
                                                                    # Forms B-C-D-B loop
    loops = find_loops(graph, group_bcdb, min_segments=3, min_length_mi=2.9, max_length_mi=3.1) # 1+1+1=3
    assert len(loops) == 1
    assert get_edge_ids(loops[0]) == {"s2", "s4", "s8_conn"}

def test_find_loops_larger_loop_with_connectors(sample_graph_and_segments_for_loops):
    graph = sample_graph_and_segments_for_loops["graph"]
    group_abdfa = sample_graph_and_segments_for_loops["group_abdfa"] # s1,s8_conn,s7,s9_conn
                                                                    # Forms A-B-D-F-A loop
    # Lengths: s1(1) + s8_conn(1) + s7(1.4) + s9_conn(3) = 6.4
    loops = find_loops(graph, group_abdfa, min_segments=4, min_length_mi=6.3, max_length_mi=6.5)
    assert len(loops) == 1
    assert get_edge_ids(loops[0]) == {"s1", "s8_conn", "s7", "s9_conn"}

def test_find_loops_lasso_side_loop(sample_graph_and_segments_for_loops):
    graph = sample_graph_and_segments_for_loops["graph"]
    # Lasso loop part: B-C-D-B, segments sh1, sh2, and reversed dc2
    # dc2 is B->D. Reversed dc2 is D->B.
    # sh1 (B->C), sh2 (C->D), dc2.reverse() (D->B)
    # Lengths: sh1(0.7) + sh2(0.7) + dc2(1.0) = 2.4
    lasso_loop_segments = sample_graph_and_segments_for_loops["lasso_loop_segments"]

    # The trail_group_segments for find_loops should be those that form the cycle.
    # Here, sh1, sh2, and dc2 (as its nodes B, D are involved).
    # The graph needs to have these connections.
    # The sample_graph_and_segments_for_loops fixture graph includes dc2 and its reverse.

    loops = find_loops(graph, lasso_loop_segments, min_segments=3, min_length_mi=2.3, max_length_mi=2.5)

    assert len(loops) == 1, f"Expected 1 loop, found {len(loops)}"
    expected_ids = {"sh1", "sh2", "dc2"} # dc2 is used to close the loop B-C-D-B
    assert get_edge_ids(loops[0]) == expected_ids

def test_find_loops_empty_input(sample_graph_and_segments_for_loops):
    graph = sample_graph_and_segments_for_loops["graph"]
    loops_empty_group = find_loops(graph, [])
    assert len(loops_empty_group) == 0

    empty_graph = nx.DiGraph()
    group_abc = sample_graph_and_segments_for_loops["group_abc"]
    loops_empty_graph = find_loops(empty_graph, group_abc)
    assert len(loops_empty_graph) == 0

def test_find_loops_filter_by_length_and_segments(sample_graph_and_segments_for_loops):
    graph = sample_graph_and_segments_for_loops["graph"]
    group_abc = sample_graph_and_segments_for_loops["group_abc"] # Length 3.4, 3 segments

    # Too short
    loops = find_loops(graph, group_abc, min_length_mi=3.5)
    assert len(loops) == 0
    # Too long
    loops = find_loops(graph, group_abc, max_length_mi=3.3)
    assert len(loops) == 0
    # Too few segments
    loops = find_loops(graph, group_abc, min_segments=4)
    assert len(loops) == 0
    # Just right
    loops = find_loops(graph, group_abc, min_segments=3, min_length_mi=3.4, max_length_mi=3.4)
    assert len(loops) == 1

# --- Tests for is_cluster_routable ---

# Re-use node definitions for clarity
# A = (0.0, 0.0), B = (1.0, 0.0), C = (1.0, 1.0), D = (2.0, 1.0)
# E = (3.0, 1.0), F = (3.0, 0.0), G_node = (0.0, 1.0)

@pytest.fixture
def graph_for_connectivity():
    graph = nx.DiGraph()
    # Component 1: A-B-C
    s1 = Edge(seg_id="s1c", name="SegA-B", start=A, end=B, length_mi=1, elev_gain_ft=10, coords=[A,B])
    s2 = Edge(seg_id="s2c", name="SegB-C", start=B, end=C, length_mi=1, elev_gain_ft=10, coords=[B,C])
    # Component 2: D-E
    s3 = Edge(seg_id="s3c", name="SegD-E", start=D, end=E, length_mi=1, elev_gain_ft=10, coords=[D,E])
    # Connector C-D (virtual or real)
    s_conn = Edge(seg_id="s_connc", name="ConnC-D", start=C, end=D, length_mi=0.5, elev_gain_ft=5, coords=[C,D], kind="connector")
    s_virt_conn = Edge(seg_id="s_vconnc", name="VirtConnC-D", start=C, end=D, length_mi=0.1, elev_gain_ft=0, coords=[C,D], kind="virtual")

    # Isolated segment F-G_node
    s_iso = Edge(seg_id="s_isoc", name="SegF-G", start=F, end=G_node, length_mi=1, elev_gain_ft=10, coords=[F,G_node])

    all_segs = [s1, s2, s3, s_conn, s_virt_conn, s_iso]
    for seg in all_segs:
        graph.add_node(seg.start)
        graph.add_node(seg.end)
        graph.add_edge(seg.start, seg.end, edge=seg)
        if seg.direction == "both": # Assume all are 'both' for test simplicity
            graph.add_edge(seg.end, seg.start, edge=seg.reverse())

    return {
        "graph": graph,
        "s1": s1, "s2": s2, "s3": s3,
        "s_conn": s_conn, "s_virt_conn": s_virt_conn,
        "s_iso": s_iso
    }

def test_is_cluster_routable_connected_simple(graph_for_connectivity):
    graph = graph_for_connectivity["graph"]
    s1 = graph_for_connectivity["s1"]
    s2 = graph_for_connectivity["s2"]
    cluster = [s1, s2] # A-B, B-C
    assert is_cluster_routable(graph, cluster) == True

def test_is_cluster_routable_connected_via_connector(graph_for_connectivity):
    graph = graph_for_connectivity["graph"]
    s1 = graph_for_connectivity["s1"] # A-B
    s2 = graph_for_connectivity["s2"] # B-C
    s3 = graph_for_connectivity["s3"] # D-E
    s_conn = graph_for_connectivity["s_conn"] # C-D connector

    # Cluster A-B, B-C, D-E. Connected by C-D (s_conn) which is IN THE GRAPH, not explicitly in cluster_segments
    # The function is_cluster_routable builds a subgraph from nodes in cluster_segments and edges from main graph.
    cluster = [s1, s2, s3]
    assert is_cluster_routable(graph, cluster) == True

    # If s_conn was part of cluster_segments, it should also be true
    cluster_with_conn = [s1, s2, s3, s_conn]
    assert is_cluster_routable(graph, cluster_with_conn) == True

def test_is_cluster_routable_connected_via_virtual_connector(graph_for_connectivity):
    graph = graph_for_connectivity["graph"]
    s1 = graph_for_connectivity["s1"]
    s2 = graph_for_connectivity["s2"]
    s3 = graph_for_connectivity["s3"]
    # Replace the real connector C-D with a virtual one in the graph for this test scenario
    # For this, we need to modify the graph passed or ensure s_virt_conn is the one connecting C and D.
    # The fixture graph_for_connectivity already has s_virt_conn between C and D.
    # Let's assume the s_conn (real connector) is NOT in the graph for this specific test.

    graph_no_real_conn = nx.DiGraph()
    base_segs = [graph_for_connectivity["s1"], graph_for_connectivity["s2"], graph_for_connectivity["s3"], graph_for_connectivity["s_virt_conn"]]
    for seg in base_segs:
        graph_no_real_conn.add_node(seg.start)
        graph_no_real_conn.add_node(seg.end)
        graph_no_real_conn.add_edge(seg.start, seg.end, edge=seg)
        if seg.direction == "both": graph_no_real_conn.add_edge(seg.end, seg.start, edge=seg.reverse())

    cluster = [s1, s2, s3] # A-B, B-C, D-E. Graph now only has virtual C-D.
    assert is_cluster_routable(graph_no_real_conn, cluster, allow_virtual_connectors=True) == True
    assert is_cluster_routable(graph_no_real_conn, cluster, allow_virtual_connectors=False) == False

def test_is_cluster_routable_disconnected(graph_for_connectivity):
    graph = graph_for_connectivity["graph"]
    s1 = graph_for_connectivity["s1"] # A-B
    s_iso = graph_for_connectivity["s_iso"] # F-G_node (isolated)
    cluster = [s1, s_iso]
    assert is_cluster_routable(graph, cluster) == False

def test_is_cluster_routable_single_segment(graph_for_connectivity):
    graph = graph_for_connectivity["graph"]
    s1 = graph_for_connectivity["s1"]
    assert is_cluster_routable(graph, [s1]) == True

    # Single segment whose nodes might not be in graph (should be false, but build_nx_graph handles this)
    s_missing_nodes = Edge(seg_id="smiss", name="Missing", start=(10,10), end=(11,11), length_mi=1, elev_gain_ft=0, coords=[(10,10),(11,11)])
    assert is_cluster_routable(graph, [s_missing_nodes]) == False # Nodes (10,10), (11,11) are not in graph

def test_is_cluster_routable_empty_cluster(graph_for_connectivity):
    graph = graph_for_connectivity["graph"]
    assert is_cluster_routable(graph, []) == True

def test_is_cluster_routable_nodes_not_in_graph(graph_for_connectivity):
    graph = graph_for_connectivity["graph"]
    s_bad_start = Edge(seg_id="sbad1", name="BadStart", start=(99,99), end=A, length_mi=1, elev_gain_ft=10, coords=[(99,99),A])
    s_bad_end = Edge(seg_id="sbad2", name="BadEnd", start=A, end=(98,98), length_mi=1, elev_gain_ft=10, coords=[A,(98,98)])
    s_both_bad = Edge(seg_id="sbad3", name="BothBad", start=(97,97), end=(96,96), length_mi=1, elev_gain_ft=10, coords=[(97,97),(96,96)])

    # Even if segments connect to valid node A, the subgraph for connectivity
    # will only include nodes from cluster_segments. If a node like (99,99) is not in the main graph,
    # the subgraph might become disconnected or behave unexpectedly.
    # is_cluster_routable adds nodes from cluster_segments to its subgraph.
    # If these nodes are not in the main `graph`, they won't have any edges from `graph` connected to them.

    # If s_bad_start is the only segment, it's routable if (99,99) and A are in graph. A is, (99,99) is not.
    # The current implementation of is_cluster_routable for a single segment checks graph.has_node for both.
    assert is_cluster_routable(graph, [s_bad_start]) == False

    cluster1 = [s_bad_start, graph_for_connectivity["s1"]] # (99,99)-A, A-B
    # Nodes in cluster: (99,99), A, B. Subgraph will have these.
    # Edges from main graph: A-B. (99,99)-A is not in main graph.
    # So, (99,99) will be an isolated node in the subgraph.
    assert is_cluster_routable(graph, cluster1) == False

    cluster2 = [s_bad_end, graph_for_connectivity["s1"]] # A-(98,98), A-B. This forms two edges from A.
    # Nodes: A, (98,98), B. Subgraph has A-B. (98,98) is isolated.
    assert is_cluster_routable(graph, cluster2) == False

    cluster3 = [s_both_bad] # (97,97)-(96,96). Neither node in graph.
    assert is_cluster_routable(graph, cluster3) == False
