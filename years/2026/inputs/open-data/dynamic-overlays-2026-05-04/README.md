# Dynamic Planning Overlays

Created: 2026-05-04

These sources are not static route geometry inputs. Use them during scheduling
and pre-run validation.

## Ridge To Rivers Conditions

- Interactive map: https://gisprod.adacounty.id.gov/apps/r2r/
- City of Boise explainer: https://www.cityofboise.org/news/parks-and-recreation/2022/november/new-interactive-map-feature-allows-ridge-to-rivers-users-to-check-real-time-trail-conditions/

Use as `day_of_trail_status.geojson` or equivalent when an extract is available.
Fields to normalize: `trail_id_or_name`, `condition_status`, `last_updated`,
`closure_flag`, `avoid_flag`, `all_weather_flag`.

## Ridge To Rivers Closures And Advisories

- Trail news / closure example: https://www.ridgetorivers.org/trail-news/seasonal-ridge-to-rivers-trail-closures-start-in-december-to-prevent-damage-protect-wildlife-habitat/

Use as a route-validity warning layer. Do not mark a generated route field-ready
when it crosses a currently closed trail or relies on closed vehicle access.

## ACHD Roadwork / RITA

- Roadwork in the Area: https://www.achdidaho.org/my-commute/roadwork-in-the-area
- Traffic advisories: https://www.achdidaho.org/my-commute/traffic/news-alerts

Use for drive-time reliability and alternate trailhead selection. Do not replace
OSM/R2R trail routing with this data.
