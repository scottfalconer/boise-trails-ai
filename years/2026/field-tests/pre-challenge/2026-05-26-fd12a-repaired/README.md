# Pre-Challenge Test: Repaired West Climb Loop

Date: 2026-05-26
Status: analysis complete; packet artifacts regenerated
Phase: pre-challenge field test

## Planned Outing

Attempted outing:

```text
1A-2. West Climb / Full Sail
formerly FD12A during the field test
2h 6m p75 / 2h 22m p90 door-to-door at test time
3.13 official miles
4.86 on-foot miles
```

Planned trails:

- Full Sail Trail
- Bob Smylie
- Buena Vista Trail

Planned official segment count: 9.

## Actual Run

User-reported door-to-door time:

```text
1:32:23.89
```

Strava activity summary from the ignored API pull:

```text
Activity name: Lunch Run
Activity type: Run
Strava start time: 2026-05-26 12:13 local
Distance: 4.95 mi
Moving time: 1h 16m 10s
Elapsed recording time: 1h 16m 12s
Elevation gain: 762 ft
Segment efforts in detailed Strava record: 8
```

The raw Strava activity JSON and GPS polyline are intentionally not committed.

## Segment Match

This is a local geometry-match result, not official challenge credit. It uses
the current local matcher with a 0.045-mile proximity threshold, 0.85 minimum
segment sample fraction, and endpoint-proximity review.

Result against the repaired West Climb source tested on 2026-05-26:

- Matched 9 of the 9 planned official segments.
- Matched 3.13 of 3.13 planned official miles.
- No planned misses.
- No extra full-segment completions.
- Partial overlaps only: `Kemper's Ridge Trail 1`, `Kemper's Ridge Trail 3`,
  `Who Now Loop Trail 2`, and `Who Now Loop Trail 3`; these do not count as
  completed segments.

Planned segment groups that appear completed:

| Trail | Segment ids | Official mi | Result |
| --- | --- | ---: | --- |
| Buena Vista Trail | 1504, 1505, 1506, 1507, 1755 | 1.38 | matched |
| Full Sail Trail | 1565, 1566 | 0.95 | matched |
| Bob Smylie | 1718, 1719 | 0.80 | matched |

## Split-Boundary Follow-Up

`Kemper's Ridge Trail 1` (`1579`) remains unfinished by this activity, but it is
now owned by active route `FD12A` in the regenerated packet. The planner issue was
not that today's run should get credit for it; the activity only partially
overlapped it. The follow-up active card adds the full Kemper 1 out-and-back
spur and replaces the stale 11.65-mile West Climb field-day line with a compact
5.96-mile edge-cover line.

## Timing Finding

The repaired West Climb estimate was conservative.

```text
Current card: 126 min p75 / 142 min p90
Actual:        92.4 min door-to-door
```

The run finished about 33.6 minutes under p75 and about 49.6 minutes under p90.
Actual Strava distance was 4.95 mi, close to the tested 4.86 mi route-card
on-foot estimate. The active follow-up `FD12A` card is now 5.96 mi because it
adds full `1579` credit.

Do not lower the p75 from this single noon pre-challenge run. The useful
finding is that the repaired one-car loop is field-executable and the prior
long `FD12A` artifact was stale route truth.

## Artifact Finding

The activity answered two different questions before regeneration:

- Against the repaired private canonical route source, the run covered 9/9
  planned segments.
- Against the stale public/example phone-packet `FD12A` source, the same run
  would have looked like 9/21 segments with 12 misses because that artifact was
  still carrying the old Harrison / Who Now / Kemper / Hippie Shake bundle.

The fix was to regenerate public map data and the phone field packet from the
private canonical source, repair the stale FD12A source geometry, then rerun the
field-packet validation chain.

## Validation

Commands run after the artifact repair:

```bash
python years/2026/scripts/export_example_map.py
python years/2026/scripts/export_mobile_field_packet.py
python years/2026/scripts/field_progress_report.py
python years/2026/scripts/field_recertification_report.py
python years/2026/scripts/field_tool_completion_audit.py
python years/2026/scripts/field_route_walkthrough_audit.py
python years/2026/scripts/route_bridge_duplication_audit.py --report-only
python -m pytest -q
```

Validation results:

- `field_progress_report.py`: passed, remaining coverage preserved.
- `field_recertification_report.py`: passed, remaining full completion feasible.
- `field_tool_completion_audit.py`: passed 18/18 requirements.
- `field_route_walkthrough_audit.py`: passed 49/49 routes.
- `route_bridge_duplication_audit.py --report-only`: advisory bridge debt remains reportable; no graduated blocking strict bridges.
- `python -m pytest -q`: rerun with the final packet before push.

## Source Notes

Private raw source, ignored by git:

```text
years/2026/inputs/strava/api-pulls/2026-05-26-fd12a-field-test/
```

Private activity review outputs, ignored by git:

```text
years/2026/outputs/private/progress/activity-review-2026-05-26-fd12a.json
years/2026/outputs/private/progress/activity-review-2026-05-26-fd12a-stale-public-packet.json
```

Public sanitized machine summary:

```text
strava-summary.json
```
