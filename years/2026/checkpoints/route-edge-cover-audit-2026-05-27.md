# Route Edge-Cover Audit

- Status: `passed`
- Routes: 31
- Failed routes: 0
- Depot phase-reset failures: 0
- Advisory routes: 3
- Advisory depot phase resets: 3
- Missing GPX routes: 0

## Failed Routes

No hard failures.

## Advisory Routes

- 11-1: 11: [{'code': 'depot_revisit_before_required_edges_cleared', 'seq': 3, 'cue_type': 'car_pass_connector', 'remaining_segment_ids': ['1572', '1573'], 'message': 'The route revisits the depot before all required edges are credited, but every required edge in this route is ascent-constrained; keep advisory until a directed split-route replacement or stronger directed route proof improves it.', 'severity': 'advisory'}]; generated 7.31 mi / lower bound 5.68 mi
- 2-1: 2: [{'code': 'depot_revisit_before_required_edges_cleared', 'seq': 18, 'cue_type': 'car_pass_connector', 'remaining_segment_ids': ['1615', '1616'], 'message': 'The route revisits the depot before all required edges are credited, but the required-edge subgraph is disconnected; keep advisory until a same-depot replacement or connector-graph proof is concrete.', 'severity': 'advisory'}]; generated 18.8 mi / lower bound 16.94 mi
- 6-1: 6: [{'code': 'depot_revisit_before_required_edges_cleared', 'seq': 12, 'cue_type': 'car_pass_connector', 'remaining_segment_ids': ['1516'], 'message': 'The route revisits the depot before all required edges are credited, but the required-edge subgraph is disconnected; keep advisory until a same-depot replacement or connector-graph proof is concrete.', 'severity': 'advisory'}]; generated 22.41 mi / lower bound 20.33 mi
