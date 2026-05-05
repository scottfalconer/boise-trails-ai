# Boise Trails AI - System Architecture

## Overview

This document describes the architecture of the Boise Trails AI planner, designed to help participants complete the Boise Trails Challenge with minimal effort while providing clear, actionable instructions.

## Core Goals

1. **Complete 100% of segments** with least manual effort (on-foot distance + elevation change)
2. **Provide clear, usable instructions** including driving time, on-foot mileage, trailheads/parking, and trail junctions

## Architecture Components

### 1. Data Layer (`core/models.py`)

Defines the core data structures used throughout the system:

- **TrailSegment**: Represents a single trail segment with:
  - Geometry (coordinates)
  - Direction requirements (one-way, ascent-only, etc.)
  - Elevation profile
  - Official/connector status
  - Metadata (name, difficulty, surface type)

- **TrailNetwork**: Graph representation of all trails:
  - Mixed directed/undirected graph (NetworkX)
  - Node/edge attributes (elevation, distance, effort)
  - Component detection and healing
  - Path finding capabilities

- **Route**: A sequence of trail segments forming a single outing:
  - Ordered list of segments
  - Total distance, elevation gain, estimated time
  - Start/end trailhead
  - Turn-by-turn instructions

- **HikingPlan**: Complete multi-day plan:
  - Collection of routes
  - Driving instructions between trailheads
  - Daily schedule with time estimates
  - Progress tracking

### 2. Graph Processing Engine (`core/graph_engine.py`)

Handles all graph-related operations:

- **Graph Construction**:
  - Load trail data from multiple sources (GeoJSON, OSM)
  - Build mixed graph with directional constraints
  - Calculate edge weights using distance + elevation
  - Apply graph healing to connect nearby components

- **Cost Calculation**:
  - Asymmetric costs for uphill/downhill (Windy Postman)
  - Elevation-aware effort metrics
  - Time estimates based on pace and terrain

- **Path Operations**:
  - Shortest path between any two points
  - Component analysis and connectivity
  - Edge traversal tracking

### 3. Optimization Engine (`core/optimizer.py`)

Implements multiple strategies for route optimization:

- **Strategy Pattern** allows switching between approaches:
  
  - **CARPStrategy**: Vehicle Routing Problem approach
    - Multiple "vehicles" (hiking days) with capacities
    - Minimizes total distance while respecting constraints
    - Uses OR-Tools for solving
  
  - **CPPStrategy**: Chinese Postman Problem approach
    - Single continuous route covering all segments
    - Optimal for "fastest known time" attempts
    - Includes TSP for component ordering
  
  - **HybridStrategy**: Combines both approaches
    - Groups nearby segments into clusters
    - Applies CPP within clusters
    - Uses VRP between clusters

- **Optimization Features**:
  - Multi-objective (distance + elevation)
  - Constraint handling (day lengths, parking)
  - Incremental optimization for partial completion

### 4. Route Decoder (`core/route_decoder.py`)

Converts optimization results into practical hiking instructions:

- **Route Reconstruction**:
  - Transform solver output to actual trail paths
  - Identify when to drive vs. walk between segments
  - Handle route splitting at drive thresholds

- **Instruction Generation**:
  - Detect trail junctions and decision points
  - Generate turn-by-turn directions
  - Include landmarks and waypoints
  - Add safety notes for difficult sections

- **Parking Optimization**:
  - Select best trailhead for each route
  - Consider parking availability and capacity
  - Minimize walking to first trail

### 5. Output Generator (`core/output_generator.py`)

Creates user-friendly outputs in multiple formats:

- **Summary Reports**:
  - CSV with daily plan overview
  - JSON for programmatic access
  - Markdown for human reading

- **Navigation Files**:
  - GPX files with routes and waypoints
  - KML for Google Earth visualization
  - GeoJSON for web mapping

- **Detailed Instructions**:
  - PDF/HTML with maps and directions
  - Driving directions between trailheads
  - Time estimates and difficulty ratings

### 6. Command Line Interface (`cli.py`)

User-friendly interface for all operations:

```bash
# Generate optimal daily plan
trail-route plan --config config/daily_planner_config.yaml

# Create single continuous route
trail-route continuous --output-dir output/continuous

# Update plan with completed segments
trail-route update --completed segments.json --plan existing_plan.json

# Analyze route efficiency
trail-route analyze --plan output/daily_plan.json
```

## Data Flow

1. **Input**: Trail data (GeoJSON) + User config (YAML) + Completion status
2. **Graph Building**: Create network representation with costs
3. **Optimization**: Apply selected strategy to find optimal routes
4. **Decoding**: Convert solution to practical instructions
5. **Output**: Generate files and reports for navigation

## Key Design Decisions

### 1. Modular Architecture
- Clear separation of concerns
- Easy to test individual components
- Extensible for new features

### 2. Strategy Pattern for Optimization
- Support multiple solving approaches
- Easy comparison of methods
- Adapt to different user needs

### 3. Graph-Based Representation
- Natural fit for trail networks
- Efficient path operations
- Standard algorithms available

### 4. Elevation-Aware Planning
- True effort minimization (not just distance)
- Better time estimates
- Fatigue consideration

### 5. Practical Output Focus
- Real-world usability over theoretical optimality
- Clear instructions for non-technical users
- Multiple format options

## Performance Considerations

- **Caching**: Store computed paths and cost matrices
- **Parallelization**: Process independent components concurrently
- **Incremental Updates**: Replan only affected portions
- **Lazy Loading**: Load trail data on demand

## Future Enhancements

1. **Web Interface**: Interactive route planning and visualization
2. **Mobile App**: Real-time navigation and progress tracking
3. **Social Features**: Share plans and compete with others
4. **Weather Integration**: Adjust plans based on conditions
5. **ML Optimization**: Learn from user preferences and performance

## Testing Strategy

- **Unit Tests**: Each component tested in isolation
- **Integration Tests**: End-to-end planning scenarios
- **Validation Tests**: Ensure 100% segment coverage
- **Performance Tests**: Handle large trail networks
- **User Acceptance**: Real-world plan evaluation