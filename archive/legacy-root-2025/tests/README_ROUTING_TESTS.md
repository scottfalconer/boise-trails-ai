# Routing System Test Strategy

This document explains the pytest test suite designed to ensure the routing system meets its stated goals.

## Test Organization

### 1. Goal-Based Tests (`test_routing_goals.py`)

Tests are organized around the two primary goals:

#### Goal 1: Complete 100% of Segments with Minimal Manual Effort
- **Coverage Tests**: Ensure all 247 required segments are covered exactly once
- **Efficiency Tests**: Verify redundancy stays below 15%
- **Elevation Tests**: Check routes minimize unnecessary climbing (no yo-yo patterns)
- **Backtracking Tests**: Ensure segments aren't traversed unnecessarily
- **Trailhead Tests**: Verify minimal driving between parking areas
- **Distance Limit Tests**: Ensure daily distances match configured limits
- **Connector Usage Tests**: Verify connector trails create loops vs out-and-backs
- **Road Minimization Tests**: Ensure road walking is <20% and only safe roads

#### Goal 2: Provide Clear, Usable Instructions  
- **Parking Tests**: Complete parking info (coords, type, fees, notes)
- **Driving Tests**: Turn-by-turn directions between trailheads
- **Navigation Tests**: Detailed waypoints with landmarks every ~0.25 miles
- **Junction Tests**: All trail junctions have GPS waypoints
- **Segment Marking Tests**: Clear entry/exit markers for required segments
- **Escape Route Tests**: Bailout options for hikes >5 miles
- **Condition Tests**: Trail difficulty and hazard warnings
- **GPX Quality Tests**: Proper metadata, sufficient GPS points
- **Time Estimate Tests**: Reasonable estimates based on distance + elevation

### 2. Architecture Tests (`test_trailhead_approach.py`)

Tests specific to the trailhead-based routing approach:

#### Trailhead Discovery
- Finds all major known trailheads
- Validates parking against OSM data
- Correctly maps accessible trail segments

#### Network Integration
- Builds unified graph with trails + roads
- Proper edge weights (required: 0.8x, connector: 1.0x, road: 1.5x)
- Good connectivity (>90% nodes connected)
- Road safety scoring

#### Loop Generation
- Trail family clustering (groups "Dry Creek Trail 1-6")
- Efficient connector usage (<30% ratio)
- Multiple strategies (prefer trails vs roads vs distance)
- Elevation-aware routing (stay high when possible)

#### Road Integration
- Accurate driving routes (not straight-line)
- Safe road walking only
- Alternative parking options

#### Practical Constraints
- Single trailhead per day when possible
- Seasonal accessibility (high trails early)
- Popular trailhead timing (arrive early)
- Weather contingencies

## Key Test Patterns

### 1. Efficiency Validation
```python
def test_redundancy_below_threshold(self, generated_plan):
    total_required_distance = sum(s.length for s in load_required_segments().values())
    total_actual_distance = sum(
        hike.total_distance for day in generated_plan.days 
        for hike in day.hikes
    )
    redundancy_pct = ((total_actual_distance / total_required_distance) - 1) * 100
    assert redundancy_pct < 15
```

### 2. Navigation Quality
```python
def test_turn_by_turn_navigation(self, generated_plan):
    for nav_point in hike.navigation_points:
        assert nav_point.instruction is not None
        assert nav_point.landmark is not None
        assert nav_point.gps_coords is not None
        assert len(nav_point.instruction) > 5  # Not too terse
```

### 3. Network Hierarchy
```python
def test_unified_graph_construction(self):
    for _, _, data in graph.edges(data=True):
        if data['edge_type'] == 'required_trail':
            assert data['weight'] == data['distance'] * 0.8  # Preferred
        elif data['edge_type'] == 'connector_trail':
            assert data['weight'] == data['distance'] * 1.0  # Neutral
        elif data['edge_type'] == 'road':
            assert data['weight'] >= data['distance'] * 1.2  # Penalized
```

## Running the Tests

### Basic Test Run
```bash
pytest tests/test_routing_goals.py -v
pytest tests/test_trailhead_approach.py -v
```

### Test Specific Goals
```bash
# Test only efficiency goals
pytest tests/test_routing_goals.py::TestGoal1MinimalEffort -v

# Test only navigation quality
pytest tests/test_routing_goals.py::TestGoal2ClearInstructions -v
```

### Test with Real Data
```bash
# Generate a plan and test it
python -m trail_route_ai.daily_planner --output output/test_plan
pytest tests/test_routing_goals.py --plan-file output/test_plan/summary.json
```

## Continuous Integration

These tests should run on:
1. Every commit (basic validation)
2. Before releasing a new route plan
3. When updating trail data
4. When modifying routing algorithms

## Success Metrics

A passing test suite guarantees:
- ✅ 100% segment coverage
- ✅ <15% redundancy (mostly scenic connectors)
- ✅ Natural loops from trailheads
- ✅ Clear navigation with landmarks
- ✅ Safe road crossings only
- ✅ Proper time estimates
- ✅ Weather/seasonal considerations
- ✅ Achievable within 31 days

## Adding New Tests

When adding features, ensure tests cover:
1. **Efficiency impact**: Does it increase redundancy?
2. **Navigation clarity**: Are instructions still clear?
3. **Real-world usage**: Does it match how people hike?
4. **Edge cases**: Disconnected trails, bad weather, full parking

## Test Data

Tests use a combination of:
- Real trail data (Boise GeoJSON files)
- Synthetic test cases (isolated trail systems)
- Historical runs (your 2024 GPX files)
- Mock data for edge cases

This ensures tests catch both common issues and rare edge cases.