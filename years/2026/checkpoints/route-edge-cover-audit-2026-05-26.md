# Route Edge-Cover Audit

- Status: `passed`
- Routes: 49
- Failed routes: 0
- Depot phase-reset failures: 0
- Advisory routes: 2
- Advisory depot phase resets: 2
- Missing GPX routes: 0

## Failed Routes

No hard failures.

## Advisory Routes

- 101-1: FD01A: [{'code': 'depot_revisit_before_required_edges_cleared', 'seq': 7, 'cue_type': 'car_pass_connector', 'remaining_segment_ids': ['1668', '1669', '1670'], 'message': 'The route revisits the depot before all required edges are credited, but the required-edge subgraph is disconnected; keep advisory until a same-depot replacement or connector-graph proof is concrete.', 'severity': 'advisory'}]; generated 11.97 mi / lower bound 5.54 mi
- 118-1: FD18A: [{'code': 'depot_revisit_before_required_edges_cleared', 'seq': 3, 'cue_type': 'car_pass_connector', 'remaining_segment_ids': ['1597'], 'message': 'The route revisits the depot before all required edges are credited, but the required-edge subgraph is disconnected; keep advisory until a same-depot replacement or connector-graph proof is concrete.', 'severity': 'advisory'}]; generated 14.36 mi / lower bound 12.1 mi
