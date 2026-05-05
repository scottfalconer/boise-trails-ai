# 2026 Personal Planning Preferences

Initial assumptions carried forward from the 2025 planning instructions and 2024/2025 Strava challenge-window history:

- Activity type: on foot.
- Routes should start and end at the same parking/trailhead location unless explicitly marked as a special-case shuttle.
- Prefer loops over out-and-backs when efficiency is comparable.
- Keep door-to-door time inside real family/work windows; kids, school pickups, and hard stops are often stricter constraints than fitness.
- Do not add a long deadhead run only to avoid a short drive or second nearby trailhead start.
- Accept split starts or compact nearby outings when they reduce elapsed time or make the day easier to fit around family logistics.
- Minimize total on-foot miles first, then unnecessary elevation gain.
- Clearly separate official challenge miles from connector, road, and redundant miles.
- Use `15.46 min/mi` as the starter fallback moving-pace estimate from prior personal segment-performance history until replaced by the current user's own performance data.
- Public Boise-area road running is allowed, including roads without sidewalks, unless the edge is private, `access=no`, `foot=no`, physically non-existent, or a graph artifact.

Known 2025 result to beat:

- `41.82%`
- `68.90 mi`

Scott private-state defaults for current planning:

- private state file: `years/2026/inputs/personal/2026-planner-state.private.json`
- available challenge dates: `2026-06-18` through `2026-07-18`
- max weekday outing duration: `260 min` from historical p90 plus logistics
- max weekend outing duration: `180 min` from historical p90 plus logistics
- preferred rest cadence: `0` required rest days after long outings in the historical p90 profile
- trailheads to avoid: none configured
- roads to avoid on foot: private, `access=no`, `foot=no`, physically non-existent, or graph-artifact roads/paths
- acceptable drive time between same-day hikes: `20 min` starter default
- same-day trailhead transfers are acceptable when they save elapsed time versus a long connector/out-and-back and still fit hard-stop windows
- target completion level for 2026: `100%`, with lower-availability sensitivity profiles still useful for tradeoff planning

Reusable defaults for other users:

- Start from `2026-planner-state.example.json`.
- The example's pace and availability are starter defaults based on Scott's prior years, not universal truth.
- Replace origin, completed/blocked segment state, pace, availability, road/trailhead preferences, and target completion before calling an output "personalized."
