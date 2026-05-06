# Project Scratchpad & Agent Log

This document tracks the evolution of the Boise Trails AI planner, key decisions made during development, and the reasoning behind the current implementation. It's intended as a log for both human developers and future AI agents.

## 1. The Continuous Route Problem (CPP + TSP)

### Initial Goal & Problem
The initial request was to create a single, continuous route that covers all 247 official segments of the Boise Trails Challenge. This was modeled as a "one massive run" problem, where driving between trailheads was allowed.

### V1: Greedy Algorithm (Failure)
- **Approach**: A simple greedy algorithm was first implemented in `continuous_route_planner.py`. It would find the closest uncompleted segment and add it to the route.
- **Outcome**: This produced a route, but it was highly inefficient (~343 miles with ~74 miles of "connector" travel) and contained logical flaws, such as "teleporting" between disconnected trailheads without accounting for the travel between them.
- **Key Learning**: A simple greedy approach is insufficient for a complex routing problem like this. It fails to consider the global structure of the trail network.

### V2: Chinese Postman Problem (CPP)
- **Insight**: The user correctly identified the on-foot portion of the problem as an instance of the **Chinese Postman Problem** (or more accurately, the **Mixed Chinese Postman Problem** due to one-way trails). The goal is to find the shortest path that traverses every required edge in a graph.
- **Implementation**:
    - We implemented a self-contained CPP solver in `continuous_route_planner.py` using the `networkx` library.
    - This involved building a graph of the trail network, identifying nodes with an odd degree, finding a minimum-weight perfect matching between them to calculate necessary backtracking, and then finding the Eulerian circuit.
- **Bug Fixes**:
    - **`unhashable type: 'TrailSegment'`**: Resolved by using segment IDs (which are hashable) in sets instead of the `TrailSegment` objects themselves.
    - **`NetworkXNoPath` Error**: The initial component detection was too simplistic. This was fixed by implementing a more robust `find_connected_components` function that builds a graph from *all* coordinates in a segment, ensuring that trails that touch mid-segment are correctly grouped.
    - **Eulerian Warning**: Fixed by having the CPP solver re-calculate connected subgraphs internally, making it robust to minor data gaps.

### V3: Traveling Salesperson Problem (TSP)
- **Insight**: After solving the CPP for each of the 41 disconnected trail components, we were left with a classic **Traveling Salesperson Problem**: in what order should we visit these 41 "cities" to minimize the total driving distance?
- **Implementation**:
    - The `python-tsp` library was added.
    - The script was refactored to first solve the CPP for all components, then calculate the geometric center of each component.
    - A distance matrix was created using the haversine distance between these centers.
    - The TSP solver was used to find the optimal visiting order.
- **Outcome**: This was a major success, **reducing the driving distance from ~219 miles to ~60 miles**, proving the two-stage CPP -> TSP model to be highly effective.

## 2. The Daily Planning Problem (CARP/VRP)

### User Story & Formal Definition
- **Goal**: Generate a minimum-mileage, multi-day hiking plan that respects daily time/distance limits and other constraints.
- **Formal Definition**: The user provided a professional breakdown, identifying the problem as a **Capacitated Windy Rural Postman Problem on a mixed graph**. This became the guiding framework for the new planner.

### Phased Implementation (`daily_planner.py`)

#### Phase 1: Scaffolding
- A new script, `src/trail_route_ai/daily_planner.py`, was created to house the new logic.
- The `ortools` library from Google was added to solve the Vehicle Routing Problem (VRP).
- A `daily_planner_config.yaml` was created to hold VRP-specific parameters like trailhead locations and daily capacities (e.g., short, medium, long days).

#### Phase 2: Graph Construction (RPP/MCPP/WPP)
- The core data loading functions were built to:
    - Load *all* trail segments (from `Boise_Parks_Trails_Open_Data.geojson`) and mark them as optional (`required=False`).
    - Load the *required* challenge segments (from `GETChallengeTrailData_v2.json`) and mark them as `required=True`. This implements the **Rural Postman Problem (RPP)** aspect.
    - Build a `networkx.DiGraph` to handle one-way trails, satisfying the **Mixed Chinese Postman Problem (MCPP)** requirement.
    - Use a DEM file to calculate elevation gain for each arc and create a "cost" based on `distance + (beta * elevation_gain)`, fulfilling the **Windy Postman Problem (WPP)** aspect.
- **Bug Fix**: Handled `ValueError: too many values to unpack` by making the GeoJSON parser robust to multi-part line geometries.

#### Phase 3: VRP Pre-computation
- The problem was transformed for the VRP solver by creating a "line graph" representation.
- An all-pairs shortest path cost matrix was computed, storing the travel cost (effort) between every pair of required arcs and depots. This is the most computationally intensive step.

#### Phase 4: VRP Solving
- The OR-Tools `RoutingModel` was used to solve the Capacitated Arc Routing Problem (CARP).
- The model was configured with the number of vehicles (available days), their capacities, and the cost matrix.
- **Bug Fix**: A `segmentation fault` was fixed by ensuring the `manager` and `routing` objects created in `solve_vrp` were the same ones passed to the solution printing/decoding functions.

#### Phase 5: Decoding and Output Generation
- The raw VRP solution was decoded back into a series of independent on-foot loops.
- **Key Decision**: A single VRP route could contain multiple on-foot loops separated by long connectors. The logic was enhanced to detect these "drives" (using the `drive_threshold_miles` config) and split the VRP route into separate, actionable hiking loops.
- **Key Decision**: The output `summary.csv` was enhanced to be human-readable, with a full description of the trails covered and the most logical starting trailhead for each loop, determined by proximity.
- **Bug Fixes**:
    - **`NetworkXNoPath`**: Gracefully handled cases where the trail network is disconnected by wrapping the pathfinding in a `try...except` block, correctly triggering a route split.
    - **`KeyError` and `I/O operation on closed file`**: Fixed several minor bugs related to dictionary keys and file handling during the final output generation.

### Final Outcome
The project now contains two powerful planners:
1.  `continuous_route_planner.py`: Solves for the most optimal single-run route.
2.  `daily_planner.py`: Solves the much more complex daily planning problem, producing a set of efficient, practical, and well-documented loops for completing the challenge over multiple days.

## 3. The Trailhead-Based Approach (Natural Hiking Patterns)

### Problem with VRP Approach
- **Analysis Results**: The VRP-based `daily_planner.py` produces routes with **232% redundancy** (561 miles total vs 169 official)
- **Key Issue**: Treats each segment as an isolated "delivery" rather than recognizing natural trail systems
- **Example**: Creates 247 individual hikes (one per segment) instead of ~30 efficient loops

### New Architecture: TrailheadRouter
- **Core Insight**: Mimic how hikers actually use trail systems - start from parking areas, follow connected trails, return to car
- **Implementation**: Created `trailhead_router.py` with modular architecture
- **Key Components**:
  - Trailhead discovery from OSM and trail data
  - Three-layer network: required trails + connector trails + roads
  - Trail family grouping (e.g., "Dry Creek Trail 1-8" = one system)
  - Loop generation using connector trails

### Implementation Progress & Issues

#### Phase 1: Architecture Design ✅
- Created comprehensive architecture documents (ARCHITECTURE.md, TRAILHEAD_ROUTING_APPROACH.md)
- Designed test suite to enforce <20% redundancy and usability requirements
- Expected: ~30-35 hikes with 10-20% redundancy

#### Phase 2: Initial Implementation ✅
- Successfully groups 99 trail families into 34 combined systems
- Reduces theoretical hikes from 247 to ~34 (86% reduction)
- Added progress tracking and performance optimizations

#### Phase 3: Critical Issues Discovered ❌
**Latest Analysis (2024-12-23)**:
- **Actual redundancy: 44.3%** (target: <20%)
- **Connector miles: 0** (should be >0 to create loops)
- **Total hikes: 64** (target: ~30)
- **Day 12 has 29 hikes alone!**

**Root Causes Identified**:
1. **Connector trails not being used** - Only "Road Connection" segments used, not actual trail connectors
2. **`_find_optimal_traversal()` too simplistic** - Just connects segments in order, doesn't create loops
3. **No actual CPP implementation** - Despite the architecture, core graph algorithms missing
4. **Fallback loops creating individual hikes** - Each orphaned segment becomes separate hike
5. **Statistics bug** - Roads excluded from connector_miles calculation

### Key Decisions & Next Steps

**Decisions Made**:
1. VRP approach fundamentally flawed for this problem - treats trails as deliveries
2. Trailhead-based approach is correct architecture - matches real hiking patterns
3. Must use connector trails from Boise Parks data (500+ available)
4. Need proper graph algorithms (CPP) not just greedy connections

**Required Fixes**:
1. Implement real CPP solver for trail families (like in continuous_route_planner.py)
2. Fix connector trail integration - actually use them in routes
3. Limit fallback loops - group orphaned segments better
4. Calculate driving distances between trailheads
5. Enforce 1-2 trailheads max per day

**Current Status**: Architecture is sound but core routing algorithms need proper implementation to achieve efficiency targets.

### Performance Metrics Comparison

| Metric | VRP Approach | Current Trailhead | Target |
|--------|--------------|-------------------|---------|
| Total On-foot Miles | 561.2 | 287.5 | ~195 |
| Redundancy % | 232.1% | 44.3% | <20% |
| Total Hikes | 41 | 64 | ~30 |
| Connector Trail Usage | N/A | 0 miles | >20 miles |
| Average Hike Length | 13.7 mi | 4.5 mi | 6-8 mi |
| Unique Trailheads | N/A | 28 | ~10-15 |

### Technical Debt Log

1. **Missing CPP Implementation**: The `_find_optimal_traversal()` method needs to implement actual Chinese Postman Problem solving for each trail family subgraph
2. **Connector Trail Loading**: The system loads connector trails but doesn't effectively use them in route planning
3. **Driving Distance Calculation**: No haversine distance calculation between different trailheads
4. **Day Organization**: Current bin-packing approach creates too many days with single hikes
5. **Graph Algorithms**: Need to port the working CPP solver from `continuous_route_planner.py`

### Lessons Learned

1. **Domain Knowledge Matters**: Understanding how hikers actually use trail systems (trailhead → loops → return) is crucial
2. **Graph Theory is Essential**: This is fundamentally a graph problem requiring proper algorithms, not heuristics
3. **Test-Driven Development**: Having clear metrics (redundancy %, connector usage) helps identify issues quickly
4. **Incremental Progress**: Even with correct architecture, implementation details determine success
5. **Data Quality**: The Boise Parks data with 500+ connector trails is key to efficiency - must be utilized

## 4. Detailed Structural Analysis (2024-12-23)

### Specific Implementation Failures

#### 4.1 Excessive Out-and-Back Legs
- **Example**: Day 6, Hike #21 - Big Springs 1 (0.43 mi out, 0.43 mi back)
- **Issue**: System doesn't use nearby connector trails to create loops
- **Impact**: Doubles the distance for single segments unnecessarily

#### 4.2 Connector Trail Accounting Bug
- **Symptom**: Road connections marked as `required: false` but `connector_miles = 0`
- **Root Cause**: Statistics calculation excludes roads from connector miles:
  ```python
  connector_miles = sum(seg.length_mi for seg in hike.segments 
                       if not seg.required and 'road' not in seg.name.lower())
  ```
- **Impact**: Roads counted as redundancy instead of zero-cost connectors

#### 4.3 Trailhead Sprawl
- **Example**: Day 5 uses 3 trailheads (Stack Rock, Rock Island West, Camel's Back)
- **Issue**: No constraint on trailheads per day
- **Impact**: 60-90 minutes of untracked driving time between hikes

#### 4.4 Missing Windy Cost Implementation
- **Current**: No directional costs or elevation preferences
- **Missing**: Trails like Whistling Pig and Sheep Camp are much harder uphill
- **Impact**: Routes may choose inefficient uphill directions

### Recommended Solution Architecture

| Fix | Implementation | Expected Impact |
|-----|----------------|-----------------|
| **Allow connector trails** | Give connector trails weight ≈ 0.1 × required trails | Redundancy drops to 15-20% |
| **Geographic clustering** | Pre-cluster segments by location, solve CARP per cluster | Reduces trailheads to 1-2 per day |
| **Windy costs** | Use `cost = distance + α×ascent` in graph edges | Prefers downhill on steep trails |
| **Driving time calculation** | Add haversine distances between trailheads to daily time | Realistic daily schedules |

### Critical Code Sections Needing Fixes

1. **`_find_optimal_traversal()`** - Currently just sorts segments, needs actual CPP solver
2. **`_add_connectors_to_family()`** - Adds connectors to graph but they're not used in routing
3. **`_calculate_summary_stats()`** - Excludes roads from connector_miles incorrectly
4. **`_organize_into_days()`** - No trailhead clustering or driving time constraints
5. **Graph edge weights** - Missing elevation-based directional costs

### Next Implementation Steps

1. **Port CPP solver** from `continuous_route_planner.py` to work on trail family subgraphs
2. **Fix connector usage** by updating edge weights (required=0.8, connector=0.1, road=0.2)
3. **Add geographic clustering** before loop generation to limit trailhead changes
4. **Implement windy costs** using DEM elevation data for asymmetric edge weights
5. **Calculate driving times** and enforce daily time limits including transitions 