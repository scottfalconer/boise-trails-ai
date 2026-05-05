# Trailhead-Based Routing Approach for Boise Trails Challenge

## Executive Summary

This document outlines a trailhead-centric approach to route planning that mimics how hikers naturally approach trail systems. Based on analysis of real-world hiking data and the current system's limitations, this approach promises to deliver more practical, efficient routes that minimize both manual effort and logistical complexity.

## Key Insights from Real-World Data

Analysis of actual hiking patterns from the 2024 challenge reveals:

1. **Natural Trail Groupings**: Successful hikes cover interconnected trail systems (e.g., "dry_creek" covered Dry Creek Trail 1, 2, 3, 6)
2. **Logical Loop Formation**: Routes like "three_bears" naturally connected Three Bears Trail, Shane's Trail, Central Ridge, and Ridge Crest segments
3. **Single Trailhead Focus**: Each activity typically starts and ends at the same parking location
4. **Progression Pattern**: Early challenge days had 9+ disconnected parts; later days consolidated to 1-3 activities

## Core Problems with Current VRP Approach

1. **Artificial Segmentation**: Treats each trail segment as an independent delivery point
2. **Excessive Driving**: Creates multiple small hikes requiring frequent parking changes
3. **Unnatural Routes**: Generates disconnected segments that don't follow trail flow
4. **Poor Efficiency**: 42 hikes over 19 days with significant redundancy

## Proposed Trailhead-Based Architecture

### Phase 1: Trailhead Discovery and Mapping

```python
# Pseudocode for trailhead extraction
trailheads = extract_trailheads_from_data():
    - Parse AccessFrom/AccessTo fields from GeoJSON
    - Identify parking areas from trail segment endpoints
    - Cross-reference with OSM data for parking lots/access roads
    - Create trailhead nodes with:
        - Location (lat/lon)
        - Parking capacity and type (paved lot, roadside, informal)
        - Road access quality (paved, gravel, 4WD required)
        - Accessible trail segments
        - Operating hours/restrictions
```

**Key Data Sources**:
- Trail segment endpoints that appear frequently
- AccessFrom/AccessTo fields in trail data
- **OSM road network** (`idaho-latest.osm.pbf`):
  - Highway tags for road classification
  - Parking areas (amenity=parking)
  - Access roads to trailheads
  - Surface type (paved/unpaved)
- Manual curation of major parking areas
- Community knowledge of informal parking

### Phase 1.5: Complete Network Integration (Roads + All Trails)

```python
# Build unified network with roads, required trails, and connector trails
def build_integrated_network():
    # Load ALL trails from Boise Parks GeoJSON
    all_trails_graph = load_all_trails(
        file='Boise_Parks_Trails_Open_Data.geojson',
        mark_required=False  # These are connector trails
    )
    
    # Load required challenge segments
    required_segments = load_required_segments(
        file='GETChallengeTrailData_v2.json',
        mark_required=True  # These count for the challenge
    )
    
    # Merge required segments into the trail graph
    trail_graph = merge_trail_networks(all_trails_graph, required_segments)
    
    # Load road network from OSM
    road_graph = load_osm_roads(
        file='idaho-latest.osm.pbf',
        bbox=boise_area_bounds,
        road_types=['primary', 'secondary', 'tertiary', 'unclassified', 'residential']
    )
    
    # Create unified graph with ALL components
    unified_graph = nx.compose_all([trail_graph, road_graph])
    
    # Tag edges by type for routing decisions
    for u, v, data in unified_graph.edges(data=True):
        if data.get('required'):
            data['edge_type'] = 'required_trail'
            data['weight'] = data['distance'] * 0.8  # Prefer required trails
        elif data.get('trail_name'):
            data['edge_type'] = 'connector_trail'
            data['weight'] = data['distance'] * 1.0  # Neutral weight
        else:
            data['edge_type'] = 'road'
            data['weight'] = data['distance'] * 1.5  # Slightly penalize roads
    
    return unified_graph
```

**Three-Layer Network Benefits**:
1. **Required Trails** (Challenge segments): Must be covered for completion
2. **Connector Trails** (Non-required trails): Essential for creating efficient loops
3. **Roads**: Last resort for connections, but useful for loop closure

**Connector Trail Advantages**:
- Create natural loops without backtracking
- Access required segments from better angles
- Provide scenic alternatives to road walking
- Connect trail systems that would otherwise require long road walks
- Offer bail-out options in bad weather

### Phase 2: Trail System Clustering

```python
# Build natural trail groupings from each trailhead
for each trailhead:
    trail_system = TrailSystem(trailhead)
    
    # Find all segments within reasonable reach
    accessible_segments = find_segments_within_radius(
        center=trailhead,
        max_distance=10_miles,  # Reasonable hiking distance
        graph=trail_network
    )
    
    # Group by connectivity and naming patterns
    trail_families = group_by_trail_name(accessible_segments)
    # e.g., "Dry Creek Trail 1-6" forms one family
    
    # Identify natural loops
    potential_loops = find_connected_components(trail_families)
```

### Phase 3: Optimal Loop Generation with Road Connections

```python
# Generate efficient loops from each trailhead
def generate_trailhead_loops(trailhead, required_segments, unified_graph):
    loops = []
    
    # Strategy 1: Complete Trail Systems with Connectors
    for trail_family in trailhead.trail_families:
        if has_required_segments(trail_family):
            # Include connector trails for efficient loops
            expanded_family = expand_with_connectors(
                trail_family,
                unified_graph,
                max_connector_ratio=0.3  # Max 30% connector trails
            )
            loop = solve_cpp_for_family(expanded_family)
            loops.append(loop)
    
    # Strategy 2: Road-Connected Loops
    # Use safe road walking to create efficient loops
    nearby_roads = find_walkable_roads_near_trails(
        trailhead.accessible_segments,
        unified_graph,
        max_road_distance=0.5  # miles
    )
    
    for road_segment in nearby_roads:
        if road_creates_efficient_loop(road_segment):
            loop = build_loop_with_road_connection(
                trail_segments=connected_segments,
                road_segment=road_segment,
                graph=unified_graph
            )
            loops.append(loop)
    
    # Strategy 3: Multi-Access Loops
    # Utilize multiple trail access points from roads
    alternate_access_points = find_road_access_points(
        required_segments,
        unified_graph,
        max_distance_from_trailhead=3.0  # miles
    )
    
    for access_point in alternate_access_points:
        if creates_better_loop(access_point, trailhead):
            loop = build_multi_access_loop(
                primary_trailhead=trailhead,
                secondary_access=access_point,
                segments=reachable_segments
            )
            loops.append(loop)
    
    # Strategy 4: Elevation-Optimized with Road Descents
    # Use roads for safe, quick descents after ridge traverses
    ridge_segments = filter_by_elevation(required_segments, min_elev=5500)
    valley_roads = find_valley_roads(unified_graph)
    
    for ridge_group in group_connected_ridges(ridge_segments):
        descent_road = find_best_descent_road(ridge_group.end_point, valley_roads)
        if descent_road:
            loop = build_ridge_traverse_with_road_descent(
                ridge_segments=ridge_group,
                descent_road=descent_road,
                return_to=trailhead
            )
            loops.append(loop)
    
    return loops
```

### Phase 4: Multi-Day Planning

```python
# Organize loops into daily plans
def create_daily_plan(all_loops, constraints):
    # Sort trailheads by segment density
    trailheads_by_priority = sort_by_segment_count(trailheads)
    
    daily_plans = []
    for day in available_days:
        # Select trailhead for the day
        trailhead = select_trailhead_for_day(
            remaining_segments,
            day_type=constraints[day].type,  # short/medium/long
            previous_day_location=daily_plans[-1].end_location if daily_plans else None
        )
        
        # Generate day's activities from single trailhead
        days_loops = select_loops_for_capacity(
            trailhead.available_loops,
            capacity=constraints[day].capacity
        )
        
        daily_plans.append(DayPlan(
            trailhead=trailhead,
            loops=days_loops,
            driving_time=calculate_drive_from_previous(...)
        ))
    
    return daily_plans
```

## Implementation Recommendations

### 1. Trailhead Prioritization

**Primary Trailheads** (high segment density):
- Camel's Back Park: Access to Central Ridge, Hull's Gulch systems
- Military Reserve: Table Rock, Central Ridge connections
- Stack Rock: Dry Creek, Around the Mountain access
- Bogus Basin: High elevation trails, ridge systems

**Secondary Trailheads** (specialized access):
- Harrison Hollow
- Corrals
- Polecat Reserve
- Hidden Springs

### 2. Loop Construction Principles

**Natural Flow**:
- Follow trail naming conventions (complete "Dry Creek Trail 1-6" together)
- Respect natural connectivity (ridge lines, valley systems)
- Minimize elevation yo-yo (stay at elevation when possible)

**Practical Constraints**:
- Maximum loop distance: 15-20 miles for long days
- Minimum loop efficiency: 80% required segments
- Drive threshold: Only change trailheads when >30 minutes saved

### 3. Segment Grouping Strategies

**By Trail System** (Highest Priority):
```
Three Bears System:
- Three Bears Trail 1, 2, 3, 4, 5
- Shane's Trail 1, 2, 3
- Shane's Connector
```

**By Elevation Band**:
```
High (>6000ft): Ridge trails, Bogus Basin area
Mid (4500-6000ft): Hillside traverses
Low (<4500ft): Valley trails, connectors
```

**By Difficulty**:
```
Technical: Rocky, steep trails grouped together
Moderate: Standard hiking trails
Easy: Wide, gentle paths for recovery days
```

### 4. Output Enhancement with Road Integration

**Navigation Instructions**:
```yaml
Day 5 - Military Reserve Trailhead
Parking: Main lot off Mountain Cove Rd (arrive before 8am on weekends)
Driving Directions: From downtown, take State St north, right on Veterans Memorial Pkwy, 
                   left on Mountain Cove Rd, parking on right
Total Distance: 12.4 miles (8.7 required, 2.5 connector trails, 1.2 road walking)
Elevation Gain: 2,850 ft
Estimated Time: 4.5 hours

Route Description:
1. From parking, head north on Central Ridge Trail
2. At junction with Shane's Trail (0.8mi), turn right
3. Complete Shane's Loop clockwise (segments 1, 2, connector)
4. Return to Central Ridge, continue north
5. At Three Bears junction (2.1mi), turn left
6. Follow Three Bears Trail through all segments
7. Exit to Collister Dr via neighborhood connector (mile 9.8)
8. Walk south on Collister Dr sidewalk for 0.6 miles
9. Turn left on Hill Rd, walk 0.6 miles to return to parking

Key Junctions:
- Mile 0.8: Central Ridge / Shane's junction (43.xxx, -116.yyy)
- Mile 2.1: Three Bears junction (43.xxx, -116.yyy)
- Mile 5.4: Ridge Crest intersection (43.xxx, -116.yyy)
- Mile 9.8: Exit to Collister Dr (residential area)

Road Walking Notes:
- Collister Dr: Wide sidewalk, moderate traffic
- Hill Rd: Bike lane available, uphill grade

Water Sources: None on trail, fill at trailhead
Exposure: Full sun after mile 3
Cell Service: Good on road sections, spotty on trails
```

**Enhanced Driving Logistics**:
```yaml
Driving Between Trailheads:
From: Camel's Back Park
To: Military Reserve
Distance: 4.2 miles (11 minutes)
Route: 
  1. Exit parking lot, turn right on 13th St
  2. Left on Heron St
  3. Right on 15th St  
  4. Left on Hill Rd
  5. Right on Mountain Cove Rd
  6. Parking lot on right

Traffic Notes: 
- Avoid 15th St during school hours (7:30-8:30am, 2:30-3:30pm)
- Hill Rd can be congested on weekend mornings

Alternative Parking:
- If main lot full: Overflow parking on Reserve St (0.3 mi walk)
- Street parking on Mountain Cove Rd (check signs)
```

### 5. Progressive Optimization

**Week 1-2**: Focus on easily accessible, well-connected trail systems
**Week 3-4**: Target remote or challenging segments requiring special effort
**Final Days**: Clean up any missed segments with targeted strikes

## Example: Using Connector Trails Effectively

### Without Connector Trails (Current Problem)
```
Required: Three Bears Trail 1, 2, 3
Problem: These segments don't form a loop
Current Solution: Out-and-back, covering each segment twice
Result: 6 miles total for 3 miles of required trails (100% redundancy)
```

### With Connector Trails (Proposed Solution)
```
Required: Three Bears Trail 1, 2, 3
Connectors: Lower Hull's Gulch (0.5 mi), 8th Street Trail (0.8 mi)
Solution: Three Bears 1→2→3 → 8th Street → Lower Hull's → back to start
Result: 4.3 miles total for 3 miles of required trails (43% redundancy)
Savings: 1.7 miles and avoid repetitive out-and-back
```

### Real-World Example from Your Data
Your "three_bears" run successfully used connectors:
- Required: Three Bears Trail 1-3, Shane's Trail 1-2, Ridge Crest segments
- Connectors: Central Ridge Trail (connected everything into one efficient loop)
- Result: Natural flow without backtracking

## Expected Outcomes

### Efficiency Improvements

**Current System**:
- 42 hikes over 19 days
- Significant driving between hikes
- ~30% redundant mileage
- Many out-and-back sections

**Trailhead-Based Approach with Connectors**:
- 25-30 total hikes
- 1-2 hikes per day maximum
- <15% redundant mileage (mostly connector trails)
- Natural loops using scenic trails instead of roads
- Minimal mid-day driving

### User Experience Benefits

1. **Clear Planning**: "Today I'm hiking from X trailhead"
2. **Simple Logistics**: One parking spot per day
3. **Natural Routes**: Follow actual trail connections
4. **Flexible Execution**: Can adjust loop on-trail if needed

## Technical Implementation Notes

### Graph Modifications

1. **Add Virtual Trailhead Nodes**: Create nodes representing parking areas
2. **Weight Adjustments**: Heavily weight edges that cross between trail systems
3. **Directional Constraints**: Respect one-way trails in loop construction

### Algorithm Selection

- **Within Trail System**: Use CPP for optimal coverage
- **Road Integration**: Modified Dijkstra considering:
  - Road walkability (sidewalks, bike lanes, shoulders)
  - Traffic levels by time of day
  - Safety ratings from OSM tags
- **Between Systems**: Route planning using actual road network
- **Daily Planning**: Constraint satisfaction with preferences

### Performance Optimizations

1. **Precompute Trail Families**: Cache which segments belong together
2. **Distance Matrix**: Precompute trailhead-to-segment distances
3. **Parallel Processing**: Generate loops for each trailhead independently

## Validation Metrics

1. **Coverage**: 100% of required segments completed
2. **Efficiency**: Total distance < 110% of minimum theoretical
3. **Practicality**: Average hikes per day ≤ 2
4. **Accessibility**: Each hike starts from designated parking

## Risk Mitigation

**Disconnected Segments**: Some segments may not connect well to any trailhead
- Solution: Create special "cleanup" days for outliers

**Parking Limitations**: Popular trailheads may be full
- Solution: Provide alternate parking options and early arrival recommendations

**Weather/Seasonal Access**: Some high elevation trailheads close seasonally
- Solution: Front-load these segments early in challenge

## Meeting the Stated Goals

### Goal 1: Complete 100% of Segments with Minimal Manual Effort

**Minimizing On-Foot Distance**:
1. **Eliminate Redundant Traversals**: By completing entire trail systems in one go, we avoid repeatedly accessing the same connector trails
2. **Strategic Sequencing**: Start with high-density areas where many required segments cluster together
3. **Efficient Connectors**: Use shortest paths between trail systems, leveraging both official trails and approved road walking

**Minimizing Elevation Change**:
1. **Elevation Banding**: Group trails by elevation to avoid unnecessary climbing
2. **Directional Optimization**: Plan routes to maximize downhill on steep segments
3. **Ridge Traverses**: When at elevation, complete all ridge trails before descending
4. **Smart Scheduling**: Do high-elevation trails early in the season when temperatures are cooler

**Specific Recommendations**:
- Use the continuous route planner WITHIN each trail system (local CPP optimization)
- Implement "gravity-assisted" routing where possible (climb once, traverse, descend)
- Cache elevation profiles for all segments to enable true effort minimization
- Consider time-of-day for elevation (morning climbs, afternoon descents)

### Goal 2: Provide Clear, Usable Instructions

**Enhanced Navigation Output**:
```yaml
Pre-Hike Checklist:
- Parking: Military Reserve Main Lot (GPS: 43.6234, -116.1789)
- Parking Fee: $5 (pay station accepts card)
- Bathroom: At trailhead
- Water: Fill at fountain near entrance
- Cell Service: Good at trailhead, spotty on trails

Turn-by-Turn with Landmarks:
1. From parking lot, follow paved path past information kiosk
2. At trail fork (50 yards), bear LEFT onto Central Ridge Trail
3. Pass yellow gate (0.2 mi) - trail becomes dirt here
4. At bench viewpoint (0.5 mi), stay STRAIGHT (ignore left fork)
5. Major junction with trail sign (0.8 mi) - Turn RIGHT onto Shane's Trail

Trail Surface Conditions:
- Mile 0-1: Wide dirt road, easy walking
- Mile 1-3: Single track, some loose rocks
- Mile 3-4: Steep switchbacks, use poles if you have them

Escape Routes:
- Mile 2.5: Bail out option via Crestline Trail (saves 3 miles)
- Mile 4.0: Shortcut back via 8th Street Trail (saves 2 miles)
```

**Critical Information Architecture**:

1. **Pre-Hike Planning**:
   - Driving directions to trailhead
   - Parking logistics (fee, capacity, restrictions)
   - Trail conditions and recent reports
   - Weather considerations

2. **On-Trail Navigation**:
   - GPS waypoints at all decision points
   - Landmark-based directions ("past the big pine tree")
   - Distance/time to next junction
   - Escape routes for emergencies

3. **Segment Tracking**:
   - Clear indication when entering/exiting required segments
   - Progress indicators ("Segment 5 of 12 for today")
   - Confirmation of completion

4. **Safety Information**:
   - Water sources (or lack thereof)
   - Exposure warnings
   - Technical difficulty sections
   - Emergency contact info

**Technology Integration**:
- Export to GPX with waypoints and custom notes
- Generate QR codes linking to online maps
- Provide offline-capable HTML files
- Integration with AllTrails/Gaia GPS formats

**Multi-Format Delivery**:
1. **Quick Reference Card** (one page per hike):
   - Key stats (distance, elevation, time)
   - Simplified map with numbered waypoints
   - Emergency info

2. **Detailed Guide** (comprehensive):
   - Full turn-by-turn directions
   - Elevation profile
   - Segment checklist
   - Photos of key junctions

3. **Digital Integration**:
   - GPX files with embedded waypoints
   - Mobile-friendly web version
   - Downloadable offline maps

## Implementation Priority

To achieve both goals effectively, prioritize development in this order:

1. **Trailhead Database**: Build comprehensive list of parking areas with metadata
2. **Trail Clustering Algorithm**: Group segments by natural systems
3. **Elevation-Aware Routing**: Implement true effort minimization
4. **Navigation Generator**: Create clear, landmark-based directions
5. **Multi-Format Output**: Support various delivery methods
6. **Validation Suite**: Ensure 100% coverage and instruction clarity

## Key Advantages of OSM Integration

### 1. Accurate Distance Calculations
- **Driving**: Real road distances between trailheads, not straight-line approximations
- **Walking**: Actual sidewalk/road distances for connector segments
- **Total Time**: More accurate estimates including driving, parking, and walking

### 2. Safety Improvements
- Identify roads with sidewalks or bike lanes for safe walking
- Avoid high-speed roads without shoulders
- Find pedestrian crossings and traffic signals
- Route through neighborhoods vs highways when possible

### 3. Loop Creation Opportunities
- Use road walks to create loops instead of out-and-backs
- Connect disconnected trail systems via safe road routes
- Leverage neighborhood cut-throughs and paths
- Identify informal trail access points from residential streets

### 4. Better Logistics
- Find all available parking options near trails
- Identify seasonal road closures
- Provide accurate driving directions
- Locate facilities (bathrooms, water, stores) along routes

## Implementation Requirements

### Data Processing
```python
# OSM data extraction for Boise area
osm_data = {
    'roads': extract_highways(tags=['primary', 'secondary', 'residential']),
    'parking': extract_amenities(tags=['parking']),
    'paths': extract_highways(tags=['footway', 'cycleway', 'path']),
    'crossings': extract_nodes(tags=['crossing']),
    'facilities': extract_amenities(tags=['toilets', 'drinking_water'])
}

# Road safety scoring
def score_road_walkability(road_segment):
    score = 0
    if road_segment.has_sidewalk: score += 3
    if road_segment.has_bike_lane: score += 2
    if road_segment.speed_limit <= 25: score += 2
    if road_segment.traffic_level == 'low': score += 1
    return score
```

### Integration Points
1. **Trailhead Discovery**: Use OSM parking areas to validate and expand trailhead list
2. **Route Planning**: Include road segments in pathfinding algorithms
3. **Safety Filtering**: Only use roads meeting minimum walkability criteria
4. **Navigation**: Provide street names and turn-by-turn directions for road sections

## Routing Priority Hierarchy

When creating loops, the system should prefer connections in this order:

1. **Required trails** - Always use when they connect naturally
2. **Connector trails** - Preferred for creating loops and accessing required segments
3. **Safe roads** (with sidewalks/bike lanes) - When trails don't connect
4. **Low-traffic roads** - Acceptable for short distances
5. **Busy roads** - Only as last resort or for very short segments

## Key Implementation Note

The Boise Parks Trails GeoJSON contains ~500+ trail segments that aren't part of the challenge but are CRITICAL for efficient routing. Examples include:

- **Central Ridge Trail**: Connects Three Bears, Shane's, and other ridge systems
- **Lower Hull's Gulch**: Links upper and lower trail networks
- **Crestline Trail**: Provides bail-out options and alternate routes
- **8th Street Trail**: Connects multiple trail systems
- **Polecat Connectors**: Link Polecat Reserve trails

Without these connectors, many required segments would need to be done as inefficient out-and-backs. By including them in the routing graph, we can create natural, flowing loops that minimize total distance while maximizing trail experience.

## Conclusion

The trailhead-based approach using THREE data sources creates the optimal routing solution:

1. **Required Segments** (GETChallengeTrailData_v2.json) - What must be completed
2. **All Trail Network** (Boise_Parks_Trails_Open_Data.geojson) - How to connect them efficiently  
3. **Road Network** (idaho-latest.osm.pbf) - Last resort connections and driving routes

By building a unified graph with all three networks and using appropriate weights (prefer required trails, use connectors freely, penalize roads), we can generate routes that:

- Complete 100% of required segments
- Minimize total distance through smart use of connector trails
- Create natural loops from established trailheads
- Provide safe road connections only when necessary
- Match how experienced hikers actually complete the challenge

The key insight: **Don't just route on required segments** - use the full trail network to create efficient, enjoyable loops that minimize redundancy and maximize the hiking experience.