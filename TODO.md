# Boise Trails Challenge Planner - TODO List

This list tracks potential future enhancements and areas for improvement for the Boise Trails Challenge Planner.

## Core Routing & Planning Logic

1.  **Advanced Intra-Cluster Route Optimization (TSP-like):**
    *   The current `plan_route` uses a greedy approach for ordering segments within a macro-cluster.
    *   *TODO: Investigate heuristics like 2-opt, a guided local search, or integration with a lightweight TSP solver for the segments within a single `plan_route` call to further reduce backtracking and improve efficiency before the "path back to start".*

2.  **Finer-grained Elevation/Difficulty in Connector Choice:**
    *   The current `estimate_time` uses a `grade_factor_sec_per_100ft`. The user story mentions factoring in technical difficulty.
    *   *TODO: Explore incorporating a more explicit "difficulty" score for trail segments (if such data is available/derivable) into the weight calculation for pathfinding, beyond just raw elevation gain.*

3.  **More Sophisticated Drive Time Estimation:**
    *   Currently uses a road graph from a user-provided GeoJSON and an average speed.
    *   *TODO: Optionally integrate with external routing APIs (e.g., OSRM, GraphHopper) for more accurate drive time estimations, potentially including traffic (if internet access is permissible).*
    *   *TODO: Allow drive time estimation to use a more complete OSM road dataset (e.g., from a local `.pbf` extract processed with `pyrosm` or `osmnx`) rather than just the potentially limited `args.roads` file (which is primarily for on-foot road connectors).*

4.  **Refined "Significantly Shorter" Logic for Road Connectors:**
    *   The `--road-threshold` parameter controls when road connectors are chosen over trail detours.
    *   *TODO: Review if the current `road_threshold` logic (prefer trail if `trail_time <= road_option_time * (1 + threshold)`) perfectly matches all desired interpretations of "significantly shorter." Consider if alternative phrasings or calculations for this decision point would be beneficial for configurability.*

## User Experience & Configuration

5.  **User Preferences for Routing:**
    *   The user story mentions preferences like “avoid busy roads” or “prefer trails even if longer.”
    *   *TODO: Design and implement a mechanism for users to specify routing preferences (e.g., via a configuration file or new command-line arguments). This would require road/trail segments to have relevant metadata (e.g., "busy," "scenic") and pathfinding algorithms to consider these weighted preferences.*

6.  **Visual Distinction of Road Connectors in GPX:**
    *   The user story suggests marking road connectors distinctly in GPX files.
    *   *TODO: Investigate adding waypoints with notes, or using GPX track segment extensions, to denote transitions to/from road connector sections if this is a high-priority feature for users. Compatibility with various GPX software should be considered.*

## Data & Performance

7.  **Performance for Very Large Datasets:**
    *   The user story mentions performance considerations for large trail networks.
    *   *TODO: For extremely large trail networks or numbers of segments, profile performance of graph building, clustering, and pathfinding. Investigate optimizations like pre-calculating all-pairs shortest paths for key nodes, or using more advanced graph algorithms/data structures if bottlenecks appear.*

8.  **Handling of Specific Trail Types/Restrictions (e.g., for Biking):**
    *   The user story mentions mode-specific rules (e.g., bike-legal connectors).
    *   *TODO: If the challenge or tool needs to support multiple modes (bike, hike, etc.), segment data would need to include mode restrictions/permissions. Routing logic would then need to filter graph edges based on the selected activity mode.*

9.  **Robustness of `nearest_node` for Drive Start/End:**
    *   `estimate_drive_time_minutes` finds the nearest road graph node to activity start/end points.
    *   *TODO: Ensure that the road network used for drive time estimation is comprehensive enough around typical trail access points. If not, drives might start/end "far" from the actual trail, or paths might not be found. Consider strategies like adding specific "trailhead" nodes to the road graph if this becomes an issue.*
