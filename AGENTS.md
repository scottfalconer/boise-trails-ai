# Repository Guidelines

This project uses two trail data files under `data/traildata/`:

* `GETChallengeTrailData_v2.json` – the canonical Boise Trails Challenge dataset containing **all official segments**. Planning scripts and tests load this file by default.
* `Boise_Parks_Trails_Open_Data.geojson` – the full Boise parks trail network. This data is helpful for locating connector segments and other reference paths but is **not** required to track official challenge progress.

Keep these roles consistent so that `GETChallengeTrailData_v2.json` remains the authoritative list of required segments and the open data file continues to serve as supplemental network information.

## Testing

When modifying dataset handling or anything that parses these files, run the test suite to ensure functionality remains correct:

```bash
pytest -q
```

## Challenge Objectives and Key Metrics

When planning routes for a challenge, the primary goal is to achieve 100% completion of all specified unique segments.

### Target Challenge Statistics (Example - Boise Challenge)
- **Total Target Distance:** ~169.35 miles
- **Total Target Climb:** ~36,000 ft
- **Total Unique Segments:** 247
- **Total Unique Trails:** 100

### Planning Efficiency
While ensuring all segments are covered, the planner should aim to:
- Stay as close as possible to the target distance and elevation, without going significantly under.
- Minimize unnecessary redundant mileage and elevation gain.

### Key Evaluation Metrics
The following metrics are important for evaluating the quality of a generated challenge plan (and are included in the HTML/CSV reports):

- **Progress (Distance/Elevation) %:** Percentage of the target new official distance/elevation covered.
  - `Progress (Distance) % = (Total New Official Trail Distance / Challenge Target Distance) * 100`
  - `Progress (Elevation) % = (Total New Official Trail Elevation Gain / Challenge Target Elevation) * 100`
- **% Over Target (Distance/Elevation):** How much the total on-foot distance/elevation gain exceeds the target.
  - `% Over Target Distance = ((Total On-Foot Distance / Challenge Target Distance) - 1) * 100`
  - `% Over Target Elevation = ((Total Elevation Gain / Challenge Target Elevation) - 1) * 100`
- **Efficiency Score (Distance/Elevation):** Ratio of target to actuals, indicating how efficiently the target was met.
  - `Efficiency Score (Distance) = (Challenge Target Distance / Total On-Foot Distance) * 100`
  - `Efficiency Score (Elevation) = (Challenge Target Elevation / Total Elevation Gain) * 100`
- **Detailed Distance Breakdown:** The plan provides totals for:
  - New and Redundant Official Challenge Trail Distance
  - New and Redundant Connector Trail Distance
  - On-Foot Road Distance
  - Total On-Foot Distance
  - Total Drive Distance & Time


# Boise Trails Challenge 2025 Planning Agent

**Intent:** This agent’s purpose is to help a participant efficiently complete **all official trail segments** of the 2025 Boise Trails Challenge on foot, using the least possible extra mileage. In other words, it will generate route plans that cover every required trail segment within the challenge **while minimizing total on-foot distance**. The goal is full **100% segment completion** with an optimized, loop-based strategy so that no unnecessary distance is traveled.

## Data Sources and Definitions

To plan routes, the agent relies on three key data sources (files) that define the trails and allowable paths:

* **Official Segments List** – the upstream Boise Trails Challenge dataset (`GETChallengeTrailData_v2.json`). This file contains **all the official segments** and is loaded by the planner and tests. Some segments include a direction requirement which must be respected.

** segment completion** - GETAthleteDashboard_v2.json contains the list of completed segment in the 2025 challenge.

* **Connector Trails Network** – *Boise_Parks_Trails_Open_Data.geojson*: This is a comprehensive dataset of Boise-area trails (from the City of Boise’s open data). It includes **additional trails and paths** beyond the official challenge list. These **connector trails** can be used to link official segments into loop routes or to avoid out-and-back retracing. **However, these do NOT count toward challenge completion** – they are only used to make routes more efficient. The planner may incorporate connector trail segments to form loops or shortcuts, but it will not mark them as “completed” since they are not part of the official 2025 list.

* **Roads Data** – *idaho-latest.osm.pbf*: This is an OpenStreetMap road dataset covering Idaho. It provides **roads and urban paths** that the participant can walk on. **Road segments** can be used to connect trails or return to trailheads on foot when necessary. Using roads (e.g. walking or running along a roadside or through a neighborhood) is considered acceptable for routing purposes, though like connector trails, road mileage does **not count toward official trail mileage**. The agent can include road sections in a route if it helps create a loop or more direct connection between trail segments. Roads can also be used to drive between loops within a same geographic area assuming there is a trailhead and parking available. Time to drive / park needs to be taken into account and on-foot is preferred

**Note:** The official segments file defines the **exact trails required** for the challenge, whereas the open-data trails and roads are **supplemental options** to help weave those required segments into convenient loops. The integrity of these data sources must be maintained (i.e. `GETChallengeTrailData_v2.json` is the authoritative list of required segments, and the GeoJSON/PBF are supplemental).

## Challenge Rules & Planning Constraints

When generating the route plan, the agent must adhere to the official challenge rules and practical constraints:

* **Complete All Official Segments:** Every required trail segment in the 2025 challenge must be covered in the plan at least once. The Boise Trails Challenge is a month-long event where participants attempt to cover **every official trail segment** within the time period. The planner’s primary objective is to ensure **100% completion** of these **247 segments across 100 trails** (see Stats below). Skipping any official segment is not allowed if the goal is challenge completion.

* **Directional Segments:** Some trail segments are defined as one-way or have a specific direction of travel required for the challenge. When a segment has a direction flag in the data (for example, a segment might only count if done east-to-west but not west-to-east), the route must traverse it in that **specified direction** to count as completed. The agent should check each segment’s `direction` property and respect it. By default most segments can be done in either direction, but any marked otherwise (e.g. "direction": "oneway", "CW" for clockwise, etc.) must be followed as indicated. This ensures those segments register as completed in the challenge tracking.

* **On-Foot Travel Only:** The plan assumes the **participant is completing the challenge on foot** (hiking/running category). No bicycles or vehicles can be used to cover any trail segment for credit. All mileage in the routes will be on foot. (Driving is of course used to get to a trailhead between days, and to make a collection of routes.) 

* **Loop/Return to Start:** Each daily route should **start and end at the same location**, typically where the user parks (a trailhead or convenient parking spot). The user must be able to return to their vehicle after completing the route. This means routes are planned as **loops** or out-and-back routes rather than point-to-point. If a set of segments cannot form a perfect loop, the planner should incorporate an out-and-back or use a road/trail connector to get back to the start. One-way trips that require shuttling vehicles are not within the scope – the agent should always provide a way to get back to the starting trailhead on foot. In practice, this may involve **minimizing backtracking** by using different trails for the return whenever possible, or doing a small retrace on a segment if it’s an out-and-back trail spur.

* **Use of Connector Trails:** *Connector* (non-official) trails from the open data set can be included in the route to improve efficiency. These might help create a loop or avoid having to double back on the same path. **However, traveling on a connector trail is “extra” mileage** – it does not contribute to challenge progress except to position the runner/hiker to reach the next official segment. The agent should use connector trails judiciously: only when they help reduce overall distance or avoid excessive out-and-back repetition. Any such paths should be clearly marked in outputs (if relevant) as connector mileage so the user knows they are not official segments. The plan should aim to **minimize distance on connectors** while still benefiting from the shortcuts they provide.

* **Use of Roads:** Including road sections in a route is allowed for the sake of routing efficiency (e.g. cutting through a neighborhood on foot to connect two close-by trails, or finishing a loop via a short roadside walk). As with connectors, **road mileage is extra** and doesn’t count toward the \~169 miles of official trails. The agent should only route along roads if it significantly improves the distance or logistical ease of a loop. 

* **Challenge Timeframe:** The 2025 Boise Trails Challenge runs from **June 19 through July 19, 2025** (one month). The agent’s planning can take this into account by distributing routes across multiple days if needed. While the primary output of this agent is routing (distance optimization), it may also be used to schedule segments into daily outings within this period. The **goal is to finish all segments by the end of the challenge window**. (Participants often plan one loop per day or weekend, etc., but scheduling is flexible as long as all segments are done in time.)

* **Pace and Time Estimates:** For planning purposes, assume the user’s **average moving pace is 16 minutes per mile** on trail. This value will be used to estimate how long each route might take to hike or run. The agent can use this base pace to calculate an approximate moving time for each loop. Additionally, an adjustment for elevation gain can be applied (e.g. adding extra minutes per 100 feet of climb) to refine time estimates, since steep trails slow the pace. Using 16 min/mile as a baseline (which is a brisk hiking pace) helps ensure the planned daily routes fit within the user’s available time. Time estimates are **optional** outputs, but including them can help the user judge the difficulty of each day’s plan.

* **Minimize Total Mileage:** A core requirement is that the agent **minimizes the total on-foot distance** needed to complete all segments. The sum of all route mileage will necessarily be larger than the official 169.35 miles of trails, due to connectors, overlaps, and returns, but it should be as close to that target as possible. In other words, **redundant mileage** – any distance that is not part of a required segment (e.g. backtracking or connectors/roads) – should be kept to a minimum. The planner should seek to **avoid repeating segments** or doing extra out-and-back legs unless absolutely necessary. By intelligently ordering the segments and using connectors/roads, the agent strives to reduce “wasted” distance. This efficiency goal means the final plan’s **Total On-Foot Distance** might only be slightly above 169.35 miles (ideally the overage is small, e.g. a few percent). Keeping the extra distance low not only saves effort but also time, and it improves the user’s chance of completing the challenge within the timeframe. (Elevation gain is also a consideration – while we focus on mileage, the agent should not needlessly add climbing either. However, all official segments must be done regardless of climb, so elevation is mostly predetermined by those trails.)

Minimize unneeded elevation changes - all things being equal, attempt to cover the segments with the least amount of upward elevation changes.

## 2025 Challenge Stats & Scope

For context, here are the key statistics for the **2025 Boise Trails Challenge** on foot:

* **Total Official Segments:** 247 segments (unique trail segments that must be completed)
* **Total Trails Represented:** 100 distinct trails are covered by those segments
* **Total Official Distance:** \~169.35 miles (cumulative length of all required segments)
* **Total Elevation Gain:** \~36,000 feet (combined elevation gain across all official segments)

These figures define the scope of the challenge. A successful plan will cover approximately 169 miles of trail and 36k ft of climb in total (not all in one go, but spread over many outings). The agent’s output routes, when combined, should include *all* of that distance and climb by covering each segment. There were **247 segments** identified for 2025 – the plan must hit each of these once in the correct direction. The 100 trails number indicates many segments belong to the same larger trail (for instance, a long trail might be split into multiple segments for the challenge). The agent doesn’t necessarily need to emphasize trail names, but it’s useful to know the overall scale: tackling this challenge is roughly equivalent to doing 169 miles/36,000 ft of hiking in Boise’s foothills and beyond.

## Additional Notes (Development & Testing)

* **Strategy Summary:** The planner groups trail segments into clusters using a topology-aware approach. This method analyzes actual trail connectivity, identifies natural trail groupings (e.g., all "Dry Creek Trail" segments), and prioritizes the formation of loops, including lasso-style routes (a main trail with a side loop). This replaces a purely spatial K-Means clustering. Once these intelligent clusters are formed, the planner computes the shortest path (typically a loop or an efficient out-and-back) covering all segments within that cluster. It selects appropriate start/end trailheads for each cluster. Internally, routing algorithms like greedy ordering and 2-opt optimization are used to refine the path and minimize redundant mileage. The outcome is a set of daily routes designed to cover all required trails efficiently and logically. This document guides the agent on rules, while the implementation handles the specific clustering and routing algorithms.

* **Verification of Completion:** After generating a plan, it should be verified that all 247 required segments are accounted for. This can be done by cross-checking the segments covered in the plan against the official list. Segments not covered (or covered in the wrong direction) would indicate a planning error that needs fixing. The agent should ideally mark which segments are completed on each route for easy tracking.

* **Maintaining Data Consistency:** The roles of the files and definitions above should remain consistent in code and documentation. The file `GETChallengeTrailData_v2.json` must always reflect the official required segments for the year, and the open data GeoJSON is only used for optional connectors. If the challenge data is updated (e.g. new segments or changed directions), those changes should propagate here as well.

* **Testing:** When using or modifying this `agents.md` file (or any logic related to it) in development, **run the test suite** to ensure everything lines up. The project includes unit tests (e.g. via `pytest`) to check that route planning and data parsing behave correctly. After updating rules or data, run `pytest -q` and confirm all tests pass. This will catch issues like missing segments, incorrect distance calculations, or mis-parsed direction flags early. The content of this file is meant to assist AI agents and developers; it should be kept up-to-date with the challenge parameters so that both the AI planning logic and the tests remain in sync. Ensuring tests pass means that the agent’s knowledge (as described here) is consistent with the implementation.

By following these guidelines and constraints, the Codex-based planning agent (and any other automated tools) will have a clear understanding of the 2025 Boise Trails Challenge requirements. It can then reason about the optimal way to link all the official segments into as few and as short routes as possible, all while respecting the challenge rules. The end result will be a practical, efficient set of hiking/running routes that allow the user to conquer the Boise Trails Challenge in the most mileage-effective way!
