# Implementation Prompt for Boise Trails Route Planner

## Project Context

You are implementing a route planning system for the Boise Trails Challenge - a month-long event where participants must hike all 247 official trail segments (~169 miles) in the Boise area. The current VRP-based system creates inefficient routes with excessive driving between disconnected segments. Your task is to implement a new trailhead-based routing approach that mimics how hikers naturally complete trail systems.

## Primary Goals

1. **Complete 100% of segments with minimal manual effort** (on-foot distance + elevation change)
2. **Provide clear, usable instructions** (driving directions, on-foot mileage, trailheads/parking, trail junctions)

## Key Documents to Review

1. **`TRAILHEAD_ROUTING_APPROACH.md`** - Complete architectural design and algorithms
2. **`ARCHITECTURE.md`** - System components and data flow
3. **`ISSUES.md`** - Current problems and why new approach is needed
4. **`tests/test_routing_goals.py`** - Tests that validate Goal 1 & 2
5. **`tests/test_trailhead_approach.py`** - Tests for architecture implementation
6. **`AGENTS.md`** - Background on problem complexity (CARP, WPP, etc.)

## Data Sources

You have three critical data sources that must be integrated:

1. **Required Segments**: `data/traildata/GETChallengeTrailData_v2.json`
   - 247 segments that must be completed
   - Some have direction requirements (ascent only)
   - These are what count for challenge completion

2. **Full Trail Network**: `data/traildata/Boise_Parks_Trails_Open_Data.geojson`
   - ~500+ trail segments including non-required "connector" trails
   - CRITICAL for creating efficient loops (avoid out-and-backs)
   - Examples: Central Ridge Trail, Lower Hull's Gulch, Crestline Trail

3. **Road Network**: `data/osm/idaho-latest.osm.pbf`
   - For safe road walking connections
   - Accurate driving directions between trailheads
   - Finding parking areas and access points

## Implementation Requirements

### Phase 1: Network Building
```python
# Build unified graph with all three networks
unified_graph = build_integrated_network()
# Must include:
# - Required trails (weight = 0.8 * distance)
# - Connector trails (weight = 1.0 * distance)  
# - Safe roads only (weight = 1.5 * distance)
```

### Phase 2: Trailhead Discovery
- Extract trailheads from trail endpoints and OSM parking data
- Major trailheads: Camel's Back Park, Military Reserve, Stack Rock, Bogus Basin
- Each trailhead needs: coords, parking type, capacity, accessible segments

### Phase 3: Loop Generation
- Start from trailheads (where people actually park)
- Group segments by trail family (e.g., "Dry Creek Trail 1-6")
- Use connector trails to create natural loops
- Minimize total distance while covering required segments
- Prefer: required trails > connector trails > roads

### Phase 4: Output Generation
- Daily plans with 1-2 hikes maximum per day
- Complete parking info and driving directions
- Turn-by-turn navigation with landmarks
- GPX files with waypoints at junctions
- Distance breakdown: X miles required, Y miles connector, Z miles road

## Success Criteria (Enforced by Tests)

### Efficiency Metrics
- ✅ 100% of required segments covered
- ✅ <15% redundancy (including connector trails)
- ✅ 25-30 total hikes (not 42 like current system)
- ✅ Minimal elevation yo-yo (stay at elevation)
- ✅ <20% road walking per hike

### Navigation Quality
- ✅ Complete parking information for each hike
- ✅ Turn-by-turn directions with landmarks
- ✅ GPS waypoints at all trail junctions
- ✅ Clear segment entry/exit markers
- ✅ Escape routes for hikes >5 miles
- ✅ Time estimates based on 16 min/mile + elevation

## Example Output Structure

```yaml
Day 5 - Military Reserve Trailhead
Parking: Main lot off Mountain Cove Rd (arrive before 8am on weekends)
Total Distance: 12.4 miles (8.7 required, 2.5 connector trails, 1.2 road)
Elevation Gain: 2,850 ft
Estimated Time: 4.5 hours

Route:
1. From parking, head north on Central Ridge Trail
2. At junction with Shane's Trail (0.8mi), turn right
3. Complete Shane's Loop clockwise (required segments 1, 2, connector)
4. Return to Central Ridge, continue north
...

Key Junctions:
- Mile 0.8: Central Ridge / Shane's junction (43.xxx, -116.yyy)
- Mile 2.1: Three Bears junction (43.xxx, -116.yyy)
```

## Testing Your Implementation

1. **Run goal-based tests**:
   ```bash
   pytest tests/test_routing_goals.py -v
   ```

2. **Run architecture tests**:
   ```bash
   pytest tests/test_trailhead_approach.py -v
   ```

3. **Validate with real data**:
   - Check against `data/segment_perf.csv` for realistic loop patterns
   - Compare to successful 2024 runs in `data/results/2024/`

## Key Insights from Analysis

1. **Current Problem**: VRP solver treats each segment as independent delivery, creating 42 disconnected hikes
2. **Natural Pattern**: Real hikers complete trail families together (see "dry_creek" and "three_bears" examples)
3. **Connector Importance**: Without the ~500 connector trails, many segments require inefficient out-and-backs
4. **Road Usage**: Only use roads with sidewalks/bike lanes, prefer trails whenever possible

## Deliverables

1. **Core Implementation**:
   - Trailhead discovery module
   - Network integration (3 layers)
   - Loop generation algorithm
   - Output formatter

2. **Passing Tests**:
   - All tests in `test_routing_goals.py`
   - All tests in `test_trailhead_approach.py`

3. **Generated Plan**:
   - Complete route covering all 247 segments
   - Summary CSV and individual GPX files
   - Clear navigation instructions

## Questions to Consider

1. How will you handle disconnected trail systems that can't be linked by trails/roads?
2. What's your strategy for popular trailheads that fill up early?
3. How will you balance efficiency with user preferences (scenic vs shortest)?
4. How will you handle seasonal closures or weather contingencies?

## Getting Started

1. Review the key documents listed above
2. Set up the development environment (see README.md)
3. Run existing tests to see what needs implementation
4. Start with network building and trailhead discovery
5. Implement loop generation using the documented strategies
6. Ensure all tests pass before considering the implementation complete

Remember: The goal is to **optimize for how people actually hike** - start where they park, follow natural trail connections, and return to their car with minimal redundancy.