# Parking Access Verification

Date: 2026-05-06

Objective: promote or reject conditional parking anchors that affect the current
p90 field-day proof.

## Result

Two parking/access assumptions were promoted from conditional to source-verified
for planning:

1. Avimor Spring Valley Creek / Twisted Spring parking for the Harlow/Spring
   cluster.
2. Dry Creek / Sweet Connie roadside parking for Sweet Connie / Shingle / Sheep.

This resolves the Harlow/Spring parking blocker. It does not resolve the
Shingle Creek `1656` time blocker.

## Avimor / Harlow-Spring Cluster

Status: source-verified for planning.

Evidence:

- OSM Overpass around the Spring Creek / Avimor official segment endpoint
  returned two nearby parking features:
  - OSM way `659568503`: `amenity=parking`, `fee=no`, `capacity=36`.
  - OSM way `708686725`: `amenity=parking`, `capacity=16`.
- AllTrails Spring Valley Creek page reports `Parking: Free, Large lot` and
  recent user reports repeatedly mark parking as easy.
- Avimor's public trails page lists Spring Valley Creek, Twisted Spring,
  Ricochet, Shooting Range, Whistling Pig, Harlow Hollows, and Harlow Hollow
  Connector as trail-system entries and publishes trail rules.
- MTB Project's Hidden Springs to Spring Valley Creek route says to park near
  the Merc in Hidden Springs because it has public parking spots, supporting the
  broader public route connection from Hidden Springs into Spring Valley Creek.

Planning change:

- Added `Avimor Spring Valley Creek parking`.
- Added `Avimor Twisted Spring parking`.
- Reran the forced-anchor p90 probe.

Proof result:

- Harlow/Spring/Twisted/Whistling/Ricochet/Shooting strict field-ready coverage
  now passes under 260 minutes p90.
- Current forced-anchor checkpoint:
  `years/2026/checkpoints/p90-forced-anchor-probe-2026-05-06.md`.

## Dry Creek / Sweet Connie / Shingle

Status: source-verified for planning.

Evidence:

- SWIMBA's Shingle/Dry Creek route says parking for the route is at the Dry
  Creek Trailhead on Bogus Basin Road, across from Sweet Connie.
- Wild West Trail describes the Dry Creek to Shingle Creek loop parking as a
  gravel pull-off on Bogus Basin Road with ample parking.
- AllTrails Dry Creek - Shingle Creek Loop says parking is available at the
  trailhead and describes the loop as a same-start/same-end route.
- The user's prior challenge-window Strava-derived parking anchor is within the
  same practical start area.

Planning change:

- Promoted `Dry Creek / Sweet Connie roadside parking` to field-ready for
  planning with `source_verified_roadside_plus_strava_seen`.

Proof result:

- Sweet Connie remains solved under 260 minutes p90.
- Shingle Creek `1656` remains unsolved under the current bound:
  - best source-verified route: 292 minutes p90, 260 minutes p75,
    11.88 on-foot miles
  - best Strava-derived strict route: 293 minutes p90, 261 minutes p75,
    11.94 on-foot miles

Additional Shingle check:

- OSM Overpass also returned two parking features closer to the Shingle Creek
  lower endpoint:
  - OSM way `1173130562`: `amenity=parking`.
  - OSM way `1173130563`: `amenity=parking`.
- These were added as probe anchors and retested.
- They did not improve the route because the runnable graph access from those
  parking features is worse than the lower Dry Creek / Sweet Connie roadside
  start:
  - `Lower Shingle / Dry Creek OSM parking west`: 382 minutes p90,
    16.24 on-foot miles; graph access to the official lower endpoint is 3.91
    mi.
  - `Lower Shingle / Dry Creek OSM parking east`: 389 minutes p90,
    16.54 on-foot miles; graph access to the official lower endpoint is 4.06
    mi.
  - `Dry Creek / Sweet Connie roadside parking`: 292 minutes p90,
    11.88 on-foot miles; graph access to the official lower endpoint is 1.73
    mi.
- Do not promote those two OSM parking features as default Shingle starts unless
  a manual GPX proves a better legal connector than the current graph found.

## Sources

- https://www.avimor.com/trails-and-outdoors
- https://www.alltrails.com/trail/us/idaho/spring-valley-creek
- https://www.mtbproject.com/trail/3811378/hidden-springs-to-spring-valley-creek-avimor
- https://swimba.org/oft-shingle/
- https://wildwesttrail.co/dry-creek-to-shingle-creek-loop-boise-idaho/
- https://www.alltrails.com/trail/us/idaho/dry-creek-shingle-creek-loop
