# Parking Anchor Schema

Parking anchors are practical parked-start locations for route generation. They are not claims about capacity, legal guarantees, or current day-of access.

## Required Fields

| Field | Meaning |
|---|---|
| `facility_id` | Stable id for the anchor. |
| `facility_name` or `name` | Human-readable parked-start name. |
| `geometry` | GeoJSON point, `[lon, lat]`. |
| `has_parking` | `true` when the source supports parking/start use. |
| `parking_confidence` | Why the planner trusts it. |
| `source` | Source dataset or manual/user source. |

## Recommended Fields

| Field | Meaning |
|---|---|
| `parking_minutes` | Default prep/parking overhead. |
| `access_notes` | Short operational note for field use. |
| `vehicle_access` | `paved`, `gravel`, `dirt_road`, `roadside`, or `unknown`. |
| `seasonal_status` | `open`, `seasonal`, `unknown`, or source-specific status. |
| `day_of_check_required` | `true` when signage/gate/current access still needs a day-of check. |
| `privacy` | `public` or `private_exact_coordinates`. |

## Confidence Values

| Value | Use |
|---|---|
| `source_validated_trailhead` | Public source explicitly identifies the point as a trailhead/parking area. |
| `inferred_from_trailhead_layer` | Public source implies trailhead use but does not give strong parking detail. |
| `osm_amenity_parking_near_official_start` | OSM parking feature near an official route start. |
| `strava_reused_prior_challenge_window` | User has started/ended 3+ prior challenge-window activities here. |
| `strava_seen_prior_challenge_window` | User has started/ended 2+ prior challenge-window endpoints or activities here. |
| `strava_single_prior_challenge_window` | User has one prior challenge-window endpoint here; useful but lower confidence. |
| `manual_required` | Candidate exists, but parking/access needs explicit user review. |

## Privacy Rule

Exact Strava-derived and home-derived coordinates stay in ignored private files under `years/<year>/inputs/personal/private/`. Public summaries may include counts and confidence buckets, but not exact private coordinates or raw activity ids.
