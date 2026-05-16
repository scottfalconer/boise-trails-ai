# BTC Local Reality Requirements

Load this before planning, reviewing, ranking, repairing, or promoting a Boise Trails Challenge route. These requirements are binding route reality, not optional annotations.

## Special Trail Management

Always check current Ridge to Rivers signage, condition reports, and the interactive map before finalizing a route. At minimum, preserve these known rules:

- Lower Hulls Gulch Trail #29:
  - Odd-numbered days: downhill bike traffic only; closed to other users.
  - Even-numbered days: open to hikers and equestrians both directions, and uphill mountain bikes; closed to downhill bike traffic.
  - A route planner needs a `current_date` and user mode to evaluate legality.
- Polecat Loop Trail #81:
  - Directional for all users. Direction has changed by year; do not assume a stale direction.
  - Some short access sections have historically remained multi-directional.
- Around the Mountain Trail #98:
  - Directional; source guidance says counter-clockwise for all users, jointly managed by Ridge to Rivers and Bogus Basin.
  - Verify current year signage because Bogus-area construction and maintenance can change access.
- Deer Point / Stack Rock / Bogus Basin 2026 first-window closure:
  - Boise National Forest Order #0402-01-117 lists the Deer Point road, trail, and area closure from May 11 through June 19, 2026, in effect 6 a.m. Monday through 6 p.m. Friday each week.
  - Because the 2026 challenge starts on Thursday, June 18, treat June 18 and the relevant Friday, June 19 closure windows as legal/access blockers for affected Bogus / Stack Rock / Sweet Connie / Pat's / Eastside / Mr. Big / Freddy's / DB Connector / Boise Ridge Road / Ponderosa / Sinker Creek routing unless same-day Forest Service, Ridge to Rivers, and Bogus sources show the route is open.
  - After June 19, keep the normal day-of closure/status check; do not keep this as a season-long blocker unless a new source extends or replaces the order.
- Bucktail Trail #20A:
  - Verified source says downhill mountain bike traffic only, with uphill bike access via Central Ridge and pedestrian/equestrian accommodation via Two Point Trail.
  - Do not describe Bucktail as an odd/even pedestrian split unless current sources prove that has changed.

## Mud And Soil

Wet trail use is a hard constraint, not a preference.

- If a route would leave boot, hoof, tire, or paw prints, the trail is too wet.
- Check Ridge to Rivers daily condition reports, RainoutLine, and the interactive map before scheduling a route after rain, freeze/thaw, or snowmelt.
- If conditions are muddy, prefer non-singletrack alternatives such as the Boise Greenbelt, Boise City parks, Rocky Canyon Road, Mountain Cove Road, and Upper 8th Street Road.
- Good wet/marginal-condition bets from the Ridge to Rivers map include Dry Creek, Lower Hulls, Camel's Back trails, Toll Road, and Freestone Ridge, but still verify current reports.
- All-weather trails listed on the Ridge to Rivers map include Shoshone-Bannock Tribes Trail, Rim Trail, Harrison Hollow, Oregon Trail, upper Basalt, Red Fox, Gold Finch, Owl's Roost, Hulls Pond Loop, The Grove Loop, Red-Winged Blackbird, and Mountain Cove.
- Trails called out by Ridge to Rivers as wet/marginal-condition avoid routes include Sweet Connie, Cottonwood Creek, Old Pen, Table Rock, Polecat Loop, Big Springs, Ridgecrest, Bucktail, Central Ridge spurs, Red Cliffs, and Hidden Springs area trails.

Do not rely only on rainfall totals. Use `recent_weather`, `overnight_freeze`, `trail_condition_report`, and `soil_class` or `wet_weather_class` when available.

## Heat, Shade, And Time Of Day

- Morning routes are strongly preferred for exposed lower-foothills terrain.
- Ridge to Rivers identifies 6 a.m. to 10 a.m. as the best summer window for cooler temperatures.
- Later starts should favor shadier lower trails, stream/gulch routes when practical, or higher elevation routes toward Stack Rock and Bogus Basin.
- Bogus/Stack Rock routes may be materially cooler and more forested than town, but still require water, weather, and access checks.

Planning variables should include `start_time`, `estimated_time_by_leg`, `heat_index`, `shade_index`, `exposure_index`, and `bailout_options`.

## Water, Bailout, And Trailheads

The planner must act as a logistics assistant, not only a line generator.

- Private home/general start origins belong in ignored personal state files such as `years/<year>/inputs/personal/*private.json`.
- Treat home origins as sensitive personal data. Use them for drive-time, home-proximate trailhead, and bailout planning; do not include exact addresses in committed docs, public/shareable route outputs, research bundles, or prompts unless explicitly asked.
- Do not assume potable water exists on trail. Mark known refill points only after source or user verification.
- Candidate refill/bailout nodes to verify before relying on them: Camel's Back Park, Fort Boise/Military Reserve area, Jim Hall Foothills Learning Center area, and Bogus Basin lodge/facilities. The user does not need Bogus lodge/facilities for Bogus outings; treat them as optional amenities only, not a route blocker, unless a plan explicitly depends on lodge water/restrooms/food/staffed bailout.
- For longer or hotter outings, force explicit water planning: starting water, possible refill, bailout, and estimated time to car.
- Prefer loops that start and end at practical parking or home-proximate trailheads when that meets the user's constraints.
- Do not require shuttles unless the user explicitly allows them.
- Treat private Strava-derived parking anchors from the user's activity endpoint clusters as evidence that the user has actually parked there before. They are valid planning anchors, not theoretical suggestions and not default parking-review blockers. Still keep exact coordinates and raw activity identifiers out of public/shareable artifacts, and use a public-safe label when one is available.
- Treat user-reviewed parking anchors, including reviewed paved-road anchors, as valid planning anchors when the review decision is `yes`. Do not re-block them for generic signage/capacity review unless there is specific evidence of changed access, ambiguity, or user uncertainty.
- Legal residential road starts are acceptable when field/source checks show the road parking is public/legal, repeatable, and cue-able from the car.
- For route review and multi-start planning, assume the user can park within `0.10` mile of a public paved vehicle road when OSM/open-data shows the paved road within `0.10` mile of the relevant trailhead or official segment access endpoint. Do not count roads near only the middle/center of a segment as usable start access. Dirt roads, tracks, cat roads, service roads with unknown surface, and unpaved shoulders are usable parking only when they are known trailheads/lots or manually verified anchors.
- Do not stop the access search at the nearest mapped road. Before rejecting or promoting a road-probe route, search outward for a certifiable parking surface such as a public park, official trailhead lot, amenity parking, event meeting point, or source-described route start. A slightly farther park/trailhead anchor can be better than a closer residential road if the added connector still beats the baseline and keeps the route legal, repeatable, and cue-able.
- For Bogus Basin multi-start planning, do not promote road shoulders, service roads, or cat tracks as parking. A Bogus anchor must resolve to a known trailhead/lodge parking area or source-/field-verified public day-use parking before publication. Current known Bogus anchors are Simplot Lodge Parking Area, Nordic Lodge Parking Area, and Pioneer Lodge Parking Area.

## Family, Work, And Hard Stops

- Optimize for realistic elapsed time windows, not only fewer trailhead starts.
- Do not assume weekends have more route time than weekdays. The user's availability is date-specific and may be as open, or more open, during the week; schedule and promotion gates must use explicit availability windows and hard stops, not weekday/weekend labels.
- Do not choose a long deadhead run just to avoid a short drive or second nearby trailhead start.
- A split route is acceptable when it keeps the day inside a pickup/work window or materially reduces on-foot time, even if the route is less aesthetically pure than one big loop.
- A slower split route is still acceptable when it creates useful bailouts, mid-route car access, water/refill options, heat-risk reduction, or a cleaner family/work hard-stop plan.
- Route outputs should make hard-stop risk visible with door-to-door time, moving time, drive time, parking/prep time, and required same-day trailhead transfers.
- When a route can be done either as one long loop or two compact nearby outings, prefer the option with lower total elapsed time unless the user explicitly prioritizes trail experience for that day.

## Connectors And Roads

- Official challenge trail miles count toward progress.
- Connector trail miles, road miles, duplicate official miles, and deadhead miles do not count toward progress.
- Non-challenge "ghost" connectors can be used to link official segments without descending to roads, but label them as connector mileage.
- Road segments, including 8th Street, Bogus Basin Road, Rocky Canyon Road, or neighborhood connectors, can be used when they create a safer or more efficient loop. Label road mileage separately.
- If a route uses a named road, service road, access road, OSM path, or non-official connector, it must be named in field instructions when possible.
- The user is willing to run public roads in the Boise foothills planning area, including roads without sidewalks. Do not reject a route only because an OSM edge is `primary`, `secondary`, `tertiary`, `residential`, `service`, `track`, or similar public road class.
- Reject or block road/path connectors that are private, `access=no`, `foot=no`, physically non-existent, or graph artifacts created by bad geometry handling.
- Preserve connector provenance in outputs as classes such as `r2r_trail`, `official_repeat`, `osm_path_footway`, `osm_public_road`, or `unknown_connector`.
- Preserve multipart line geometry as separate graph parts. Never flatten a `MultiLineString` into one continuous edge for routing.
- A plan should report official new miles, official repeat miles, connector miles, road miles, total on-foot miles, drive time, elevation gain, and expected moving time.

Failure mode to catch: a route can be credit-correct but still field-wrong if the segment-order chain forces an out-and-back or repeat after its credit/access purpose is already satisfied. Do not defend the repeat as "needed for credit"; ask whether a shorter legal connector or parallel trail should replace the repeat for the remaining movement to the next cue, with elevation and direction cost included.

When the map line uses a non-credit or already-credited trail inside an official cue, label it as connector/repeat mileage in the phone cue. The active cue title, displayed GPX leg, and signed trail under the line must agree enough that a runner can tell whether they are earning new credit or simply moving legally to the next credit segment.

## Time Estimate Correctness

Time estimates are a field-safety and family/work hard-stop constraint.

- Treat `total_minutes` as the conservative door-to-door planning number shown to the user. It should be backed by `time_estimates_minutes.door_to_door_p75` when available.
- Preserve raw model output separately as `raw_total_minutes`; do not overwrite calibrated p75-style field estimates with raw segment sums.
- Every runnable outing should carry DEM-derived `effort` fields: `ascent_ft`, `descent_ft`, `grade_adjusted_miles`, `estimated_moving_minutes_p50`, and `estimated_moving_minutes_p75`.
- Route-finding complexity must be represented explicitly with a route-finding penalty or similar timing adjustment, especially where the route crosses or reuses the same trail corridor.
- Schedule p75/p90 bounds must come from explicit dated availability or the current personal profile. Do not apply a lower weekday bound or higher weekend bound merely because of the calendar label; if only coarse availability is known, report that assumption and keep placement flexible.
- Field-test outcomes should update a calibration input, not just prose notes, when actual door-to-door or moving time materially differs from the model.
- Do not promote a generated candidate as a faster or better replacement unless it has p75 time, DEM effort, and a continuous navigation GPX. Graph validation alone is not enough.
- The efficiency audit should fail if runnable cards have missing or stale p75 timing, missing DEM effort, incomplete segment coverage, or an optimizer replacement that is only faster on paper but not field-navigable.
