# Runner-Perspective Frame-Shift Audit Index

This checkpoint decomposes the current 2026 field-packet routes into start, cue/junction, and finish decision points, then reframes each point from model artifact logic into runner-on-trail logic.

Important boundary: this is a local-data visual reasoning audit. It does not contain live field photos, current Street View, or day-of Ridge to Rivers condition checks, so literal sightline claims remain proof-gated.

## Inputs

- `docs/field-packet/field-tool-data.json`
- `docs/field-packet/gpx/official/*.gpx`
- `years/2026/inputs/open-data/r2r-trails-2026-05-04/boise_parks_trails_open_data.geojson`
- `years/2026/inputs/open-data/routing-connectors-2026-05-04/combined_r2r_osm_connectors.geojson`
- `years/2026/inputs/official/api-pull-2026-05-04/official_foot_segments.geojson`

## Method

For every current field-packet route, the audit reads the route card, cue GPX waypoints, and nearby named trail/road features. It then records a frame shift at the parked start, every packet cue/decision point, and the return-to-car endpoint.

The key frame shift is: `the model sees a valid route artifact; the runner sees signs, branches, roads, other trail lines, overlap, fatigue, and uncertainty`.

## Routes

| Route | Trailhead | Cues | Access status | Audit status | File |
| --- | --- | ---: | --- | --- | --- |
| 1A-1 | Strava parking anchor 13 | 3 | private-history parking anchor; usable as planning evidence but still public-proof limited | needs_visual_proof | [1a-1-strava-parking-anchor-13.md](route-audits/1a-1-strava-parking-anchor-13.md) |
| 4B | Upper Interpretive | 3 | known-or-mapped parking in packet data | needs_visual_proof | [4b-upper-interpretive.md](route-audits/4b-upper-interpretive.md) |
| 4A | Bob's | 5 | known-or-mapped parking in packet data | needs_visual_proof | [4a-bob-s.md](route-audits/4a-bob-s.md) |
| 5A | West Hidden Springs Drive road-parking anchor | 3 | parking/access proof-sensitive road or probe anchor | needs_visual_proof | [5a-west-hidden-springs-drive-road-parking-anchor.md](route-audits/5a-west-hidden-springs-drive-road-parking-anchor.md) |
| 8B | Homestead | 3 | known-or-mapped parking in packet data | needs_visual_proof | [8b-homestead.md](route-audits/8b-homestead.md) |
| 4C-1 | Warm Springs Golf Course | 5 | parking evidence incomplete in packet data | needs_visual_proof | [4c-1-warm-springs-golf-course.md](route-audits/4c-1-warm-springs-golf-course.md) |
| 15A-2 | Bob's | 5 | parking evidence incomplete in packet data | needs_visual_proof | [15a-2-bob-s.md](route-audits/15a-2-bob-s.md) |
| 1A-2 | West Climb | 7 | parking evidence incomplete in packet data | needs_visual_proof | [1a-2-west-climb.md](route-audits/1a-2-west-climb.md) |
| 8A | Homestead | 3 | known-or-mapped parking in packet data | needs_visual_proof | [8a-homestead.md](route-audits/8a-homestead.md) |
| 7 | Seamans Gulch | 5 | known-or-mapped parking in packet data | needs_visual_proof | [7-seamans-gulch.md](route-audits/7-seamans-gulch.md) |
| 16B | Freddy's Stack Rock | 3 | known-or-mapped parking in packet data | needs_visual_proof | [16b-freddy-s-stack-rock.md](route-audits/16b-freddy-s-stack-rock.md) |
| 15B | Dry Creek Parking Area/Trailhead | 4 | known-or-mapped parking in packet data | needs_visual_proof | [15b-dry-creek-parking-area-trailhead.md](route-audits/15b-dry-creek-parking-area-trailhead.md) |
| 11 | Hawkins Range Reserve | 3 | known-or-mapped parking in packet data | needs_visual_proof | [11-hawkins-range-reserve.md](route-audits/11-hawkins-range-reserve.md) |
| 10B | Dry Creek Parking Area/Trailhead | 5 | known-or-mapped parking in packet data | needs_visual_proof | [10b-dry-creek-parking-area-trailhead.md](route-audits/10b-dry-creek-parking-area-trailhead.md) |
| 5B | Cartwright | 7 | parking evidence incomplete in packet data | needs_visual_proof | [5b-cartwright.md](route-audits/5b-cartwright.md) |
| 9 | Veterans | 10 | known-or-mapped parking in packet data | needs_visual_proof | [9-veterans.md](route-audits/9-veterans.md) |
| 19 | Cervidae / Arrow Rock Road OSM Parking | 3 | parking/access proof-sensitive road or probe anchor | needs_visual_proof | [19-cervidae-arrow-rock-road-osm-parking.md](route-audits/19-cervidae-arrow-rock-road-osm-parking.md) |
| 4C-2 | Strava parking anchor 21 | 11 | private-history parking anchor; usable as planning evidence but still public-proof limited | needs_visual_proof | [4c-2-strava-parking-anchor-21.md](route-audits/4c-2-strava-parking-anchor-21.md) |
| 15A-1 | Dry Creek / Sweet Connie roadside parking | 3 | parking/access proof-sensitive road or probe anchor | needs_visual_proof | [15a-1-dry-creek-sweet-connie-roadside-parking.md](route-audits/15a-1-dry-creek-sweet-connie-roadside-parking.md) |
| 14 | Orchard Gulch | 6 | known-or-mapped parking in packet data | needs_visual_proof | [14-orchard-gulch.md](route-audits/14-orchard-gulch.md) |
| 16A-1 | Dry Creek / Sweet Connie roadside parking | 3 | parking/access proof-sensitive road or probe anchor | needs_visual_proof | [16a-1-dry-creek-sweet-connie-roadside-parking.md](route-audits/16a-1-dry-creek-sweet-connie-roadside-parking.md) |
| 3 | Freestone Creek | 20 | known-or-mapped parking in packet data | needs_visual_proof | [3-freestone-creek.md](route-audits/3-freestone-creek.md) |
| 12 | 8th Street ATV Parking Area | 7 | known-or-mapped parking in packet data | needs_visual_proof | [12-8th-street-atv-parking-area.md](route-audits/12-8th-street-atv-parking-area.md) |
| 16A-2 | Dry Creek / Sweet Connie roadside parking | 5 | parking/access proof-sensitive road or probe anchor | needs_visual_proof | [16a-2-dry-creek-sweet-connie-roadside-parking.md](route-audits/16a-2-dry-creek-sweet-connie-roadside-parking.md) |
| 18 | Pioneer Lodge Parking Area | 11 | known-or-mapped parking in packet data | needs_visual_proof | [18-pioneer-lodge-parking-area.md](route-audits/18-pioneer-lodge-parking-area.md) |
| 2 | Hulls Gulch | 19 | known-or-mapped parking in packet data | needs_visual_proof | [2-hulls-gulch.md](route-audits/2-hulls-gulch.md) |
| 10A | Harlow's / Hidden Springs west access probe | 13 | parking/access proof-sensitive road or probe anchor | needs_visual_proof | [10a-harlow-s-hidden-springs-west-access-probe.md](route-audits/10a-harlow-s-hidden-springs-west-access-probe.md) |
| 17 | Simplot Lodge Parking Area | 11 | known-or-mapped parking in packet data | needs_visual_proof | [17-simplot-lodge-parking-area.md](route-audits/17-simplot-lodge-parking-area.md) |
| 6 | Cartwright | 10 | known-or-mapped parking in packet data | needs_visual_proof | [6-cartwright.md](route-audits/6-cartwright.md) |
| 13 | Freestone Creek | 15 | known-or-mapped parking in packet data | needs_visual_proof | [13-freestone-creek.md](route-audits/13-freestone-creek.md) |

## Cross-Route Findings

- The dominant gap is literal sightline proof: local route data can identify nearby named branches and road features, but cannot prove what the runner can visually see at trail speed.
- Routes with generic OSM connectors, private-history anchors, road/probe starts, overlap warnings, or many nearby named trails deserve the first field-photo or imagery pass.
- This audit intentionally does not promote or reject routes; it changes the proof burden from `does the route artifact exist?` to `can the runner choose correctly at each visible decision point?`.
