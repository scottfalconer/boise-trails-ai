# Cluster-Level Repricing Audit

Generated: 2026-05-12T03:55:48Z
Status: `optimized_existing_loop_clusters`

## Summary

- Routes audited: 50
- Graph edges: 77
- Multi-route components: 3
- Exact optimized components: 3
- Components with existing-loop savings: 2
- Existing-loop route cards: 40 -> 34 (6 removed)
- Existing-loop on-foot miles: 251.37 -> 237.64 (13.73 mi saved)
- Existing-loop p75 minutes saved: 536
- Existing-loop p90 minutes saved: 604

## Existing-Loop Savings Components

| Component | Routes | Segments | Saved routes | Saved on-foot mi | Saved p75 | Selected | Skipped |
|---|---:|---:|---:|---:|---:|---|---|
| C02 | 15 | 92 | 2 | 7.52 | 188 | 104-1: FD04A, 105-1: FD05A, 105-2: 4A, 106-1: FD06A, 111-1: 14, 115-1: 3, 119-1: FD19A, 119-2: FD19B, 120-1: FD20A, 122-2: FD22B, 122-3: FD22C, 123-1: 12, 128-1: FD28A | 119-3: FD19C, 122-1: FD22A |
| C01 | 19 | 56 | 4 | 6.21 | 348 | 103-1: FD03A, 108-1: FD08A, 108-2: FD08B, 109-1: FD09A, 109-2: 10B, 113-1: 16A-1, 114-2: FD14B, 116-2: 15B, 117-1: 16B, 118-1: FD18A, 124-1: FD24A, 127-2: FD27B, 128-2: 15A-1, 129-1: 16A-2, 130-1: FD30A | 114-1: FD14A, 114-3: FD14C, 127-1: FD27A, 127-3: FD27C |

## Largest Components

| Component | Routes | Edge weight | Current on-foot mi | Optimized on-foot mi | Status |
|---|---:|---:|---:|---:|---|
| C01 | 19 | 96.96 | 114.71 | 108.50 | optimized_exact |
| C02 | 15 | 57.19 | 99.92 | 92.40 | optimized_exact |
| C03 | 6 | 17.14 | 36.74 | 36.74 | optimized_exact |

## Scope Boundary

- This is a cluster-level optimizer over existing certified no-shuttle route cards.
- It is order-free cluster repricing, so promoting its skipped-route set would require a new calendar assignment and full field-packet recertification.
- It does not generate new Harlow/Avimor, Freestone/Military, Hulls/Crestline, Bogus, or Cartwright/Polecat loops.
- If a cluster still has high optimized on-foot cost, that is a candidate-universe problem: generate new no-shuttle loops for the whole component, then rerun this audit.
