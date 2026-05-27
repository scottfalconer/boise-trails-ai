# Repaired West Climb Loop Analysis

Status: complete
Updated: 2026-05-26

## Question

Did the repaired West Climb / Full Sail loop work in the field, and did the run
cover the segments in the current map?

## Short Answer

Yes for the repaired route tested that day. The local matcher found all 9 planned
official segments, with no planned misses and no extra full-segment completions.

The answer was initially ambiguous because generated public artifacts were stale:
the tested private source had the repaired 9-segment route, while the public/example
map and phone packet still exposed the old 21-segment bundle. Follow-up planning
then moved `Kemper's Ridge Trail 1` (`1579`) onto the active FD12A card. The
active packet now has a 10-segment, 5.96-mile FD12A edge-cover route with the
stale 11.65-mile West Climb source geometry removed.

This still should not be treated as official 2026 BTC credit because the
challenge window has not started.

## Evidence

The repaired field card says:

| Metric | Current card | Actual |
| --- | ---: | ---: |
| Door-to-door time | 126 min p75 / 142 min p90 | 92.4 min |
| On-foot distance | 4.86 mi | 4.95 mi |
| Official planned miles | 3.13 mi | 3.13 mi matched |
| Planned official segments | 9 | 9 matched |

The Strava run recorded:

```text
Distance: 4.95 mi
Moving time: 1:16:10
Elapsed recording time: 1:16:12
Elevation gain: 762 ft
```

The user's door-to-door time was:

```text
1:32:23.89
```

That leaves about 16m12s outside the Strava elapsed recording window.

## Segment Interpretation

Completed planned segments:

- Buena Vista Trail: 1504, 1505, 1506, 1507, 1755
- Full Sail Trail: 1565, 1566
- Bob Smylie: 1718, 1719

Partial overlaps that should not be counted:

- Kemper's Ridge Trail: 1579, 1581
- Who Now Loop Trail: 1698, 1699

These partials are useful route-history evidence only. They are not full
endpoint-to-endpoint segment completions.

## Timing Interpretation

The tested p75 appears conservative for the 9-segment repaired loop:

```text
p75 delta: -33.6 min
p90 delta: -49.6 min
distance delta: +0.09 mi
```

The tested route-card distance is close enough to the field result. The active
follow-up card is longer because it adds full `1579` credit. The timing may be
high, but a single clean sample is not enough to lower the route's p75 because
heat, fatigue, parking friction, and navigation uncertainty still matter.

## Artifact Interpretation

Before regeneration, the same activity produced two answers:

- Repaired canonical route: 9/9 planned segments complete.
- Stale public/phone `FD12A`: 9/21 planned segments complete, 12 misses.

The stale 21-segment artifact included Harrison Hollow, Harrison Ridge, Who Now
Loop, Kemper's Ridge, and Hippie Shake. Those were not part of the repaired
source the user actually tested.

The durable fix was source-first regeneration:

- reapply H1 source promotion with field-time calibrations;
- export the public/example map from private canonical source;
- regenerate the mobile field packet from canonical source;
- add the missing `15B` West Dry Creek Road start-access hint that the
  walkthrough audit surfaced during recertification;
- rerun completion, bridge, walkthrough, and pytest gates.

## Current Planning Decision

Keep active route `1A-2` in the field menu as the West Climb / Full Sail / Bob
Smylie / Buena Vista loop. The activity validates the repaired route shape as
field-executable and confirms the stale long `FD12A` artifact should not be used
for judging this run.

Do not mark official challenge progress from this run because it happened before
the 2026 challenge window.

## Split-Boundary Follow-Up

The run partially overlapped `Kemper's Ridge Trail 1` (`1579`) but did not cover
the full endpoint-to-endpoint segment, so this activity does not complete it.
The regenerated active field menu assigns `1579` to `FD12A`, not to the old
`FD12B` split, and the phone packet now draws the compact 5.96-mile repaired
edge-cover route instead of the stale 11.65-mile West Climb field-day line.

The follow-up repair was not to manually mark `1579` complete from this run. It
was to audit the active field menu for bridge duplication and cross-route tail
opportunities. The bridge audit is now clear for unwaived strict bridges, while
tail opportunities remain informational optimization work.
