# Shingle 1656 Anchor Exhaustive Probe

Objective: exhaustively test Shingle Creek 1656 against every known parking anchor

## Verdict

- Segment: Shingle Creek Trail 1 (`1656`)
- P90 bound: 260 min
- Strict success found: False
- Best field-ready anchor: Dry Creek / Sweet Connie roadside parking
- Best field-ready p90/p75: 292 / 260 min
- Minutes over bound: 32

## Summary

- Anchors tested: 74
- Graph-validated rows: 67
- Track-valid rows: 74
- Field-ready anchors: 63
- Under-bound graph/track-valid rows: 0

## Best Rows

| Rank | Anchor | P90 | P75 | On foot | Field ready | Parking | Source |
|---:|---|---:|---:|---:|---|---|---|
| 1 | Dry Creek / Sweet Connie roadside parking | 292 | 260 | 11.88 | True | source_verified_roadside_plus_strava_seen | historical_strava_anchor_plus_swimba_and_local_parking_sources |
| 2 | Strava parking anchor 23 | 293 | 261 | 11.94 | True | strava_seen_prior_challenge_window | strava_activity_endpoint_cluster |
| 3 | MillerGulch Parking Area/Trailhead | 340 | 303 | 14.76 | True | inferred_from_trailhead_layer | city_parks_facilities |
| 4 | Strava parking anchor 22 | 340 | 303 | 14.76 | True | strava_seen_prior_challenge_window | strava_activity_endpoint_cluster |
| 5 | Strava parking anchor 02 | 381 | 340 | 17.1 | True | strava_seen_prior_challenge_window | strava_activity_endpoint_cluster |
| 6 | Lower Shingle / Dry Creek OSM parking west | 382 | 341 | 16.24 | True | osm_amenity_parking_source_checked | osm_overpass_amenity_parking_2026_05_06 |
| 7 | Bob's Trailhead | 382 | 341 | 17.12 | True | inferred_from_trailhead_layer | city_parks_facilities |
| 8 | Hawkins Range Reserve Trailhead | 389 | 347 | 16.54 | True | inferred_from_trailhead_layer | city_parks_facilities |
| 9 | Lower Shingle / Dry Creek OSM parking east | 389 | 347 | 16.54 | True | osm_amenity_parking_source_checked | osm_overpass_amenity_parking_2026_05_06 |
| 10 | Strava parking anchor 09 | 389 | 347 | 16.56 | True | strava_seen_prior_challenge_window | strava_activity_endpoint_cluster |
| 11 | Strava parking anchor 15 | 389 | 347 | 17.0 | True | strava_seen_prior_challenge_window | strava_activity_endpoint_cluster |
| 12 | Cartwright Trailhead | 390 | 348 | 17.1 | True | inferred_from_trailhead_layer | city_parks_facilities |
| 13 | Upper Interpretive Trailhead | 409 | 365 | 17.6 | True | inferred_from_trailhead_layer | city_parks_facilities |
| 14 | Strava parking anchor 19 | 423 | 377 | 17.96 | True | strava_seen_prior_challenge_window | strava_activity_endpoint_cluster |
| 15 | Hard Guy Trailhead | 428 | 382 | 18.08 | True | inferred_from_trailhead_layer | city_parks_facilities |

## Best Field-Ready Rows

| Rank | Anchor | P90 | P75 | On foot | Distance basis | Distance mi | Parking |
|---:|---|---:|---:|---:|---|---:|---|
| 1 | Dry Creek / Sweet Connie roadside parking | 292 | 260 | 11.88 | start | 1.5921 | source_verified_roadside_plus_strava_seen |
| 2 | Strava parking anchor 23 | 293 | 261 | 11.94 | start | 1.5854 | strava_seen_prior_challenge_window |
| 3 | MillerGulch Parking Area/Trailhead | 340 | 303 | 14.76 | start | 2.4087 | inferred_from_trailhead_layer |
| 4 | Strava parking anchor 22 | 340 | 303 | 14.76 | start | 2.4101 | strava_seen_prior_challenge_window |
| 5 | Strava parking anchor 02 | 381 | 340 | 17.1 | start | 3.2024 | strava_seen_prior_challenge_window |
| 6 | Lower Shingle / Dry Creek OSM parking west | 382 | 341 | 16.24 | start | 0.704 | osm_amenity_parking_source_checked |
| 7 | Bob's Trailhead | 382 | 341 | 17.12 | start | 3.2089 | inferred_from_trailhead_layer |
| 8 | Hawkins Range Reserve Trailhead | 389 | 347 | 16.54 | start | 0.7696 | inferred_from_trailhead_layer |
| 9 | Lower Shingle / Dry Creek OSM parking east | 389 | 347 | 16.54 | start | 0.7737 | osm_amenity_parking_source_checked |
| 10 | Strava parking anchor 09 | 389 | 347 | 16.56 | start | 0.7817 | strava_seen_prior_challenge_window |
| 11 | Strava parking anchor 15 | 389 | 347 | 17.0 | start | 3.3925 | strava_seen_prior_challenge_window |
| 12 | Cartwright Trailhead | 390 | 348 | 17.1 | start | 3.4135 | inferred_from_trailhead_layer |
| 13 | Upper Interpretive Trailhead | 409 | 365 | 17.6 | center | 2.793 | inferred_from_trailhead_layer |
| 14 | Strava parking anchor 19 | 423 | 377 | 17.96 | start | 4.2246 | strava_seen_prior_challenge_window |
| 15 | Hard Guy Trailhead | 428 | 382 | 18.08 | center | 0.8782 | inferred_from_trailhead_layer |

## Interpretation

- This closes the straight-line-nearest-anchor loophole for Shingle 1656 by testing every known public trailhead, manual anchor, and private Strava-derived parking anchor.
- A strict success requires graph validation, continuous track validation, field-ready parking, and p90 at or below the active max bound.
- The report intentionally records anchor names and metrics, not exact private anchor coordinates.
