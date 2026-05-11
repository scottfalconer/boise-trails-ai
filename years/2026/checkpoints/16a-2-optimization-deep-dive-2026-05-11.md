# 16A-2 Optimization Deep Dive

Date: 2026-05-11

Objective: dig into `16A-2` from the runner/field-execution frame and identify
whether the optimization is a route-shortening problem, a re-anchor problem, or
a route-credit packaging problem.

## Verdict

The best optimization is not a new standalone `16A-2` start.

The current `15A-1` Dry Creek GPX already traverses official Shingle Creek
segment `1656` in the required ascent direction while only claiming Dry Creek
segments. If that holds in the eventual BTC activity recording, `16A-2` should
not spend another 14.96-mile outing to re-run Shingle. Instead:

1. Run or package `15A-1` so it claims Dry Creek plus Shingle `1656`.
2. Reduce the remaining `16A-2` work to Sheep Camp `1653` only.
3. Use the existing Sheep Camp single-segment probe as the candidate starting
   point: 3.30 on-foot miles, 106 p75 minutes, 119 p90 minutes.

That drops the post-`15A-1` `16A-2` burden from 14.96 miles / 310 p75 / 348 p90
to about 3.30 miles / 106 p75 / 119 p90, saving roughly 11.66 on-foot miles and
204 p75 minutes for the same remaining official credit.

Proof boundary: that is a conditional remaining-work reduction after `15A-1`
validates Shingle credit and the remaining menu is repriced. It is not proof
that the current full field packet has lower net human effort. The current
implemented improvement is better segment-credit provenance and auditability,
not a promoted route-card replacement.

## Current 16A-2 Diagnosis

Current route card:

- Label: `16A-2`
- Official segments: `1656` Shingle Creek Trail 1, `1653` Sheep Camp Trail 1
- Official miles: 5.53
- On-foot miles: 14.96
- Door-to-door p75 / p90: 310 / 348 minutes
- Problem shape: 9.43 non-new-credit miles, repeated Dry Creek / Shingle /
  Sheep Camp corridors, and long return/access legs.

Access correction after live OSM check:

- OpenStreetMap has an `amenity=parking` way at the Dry Creek / Sweet Connie
  start area: OSM way `1328228551`, centered at about
  `43.6916536, -116.182042`.
- That is effectively the same roadside start used by the current `15A-1`,
  `16A-1`, and `16A-2` cards, not one of the rejected lower-Shingle OSM parking
  features.
- This strengthens the parking evidence for the current start. It does not
  change the route-cost conclusion: Shingle is still expensive as a standalone
  same-car outing, and the bigger win remains claiming Shingle from `15A-1`
  before reducing `16A-2` to Sheep Camp-only.

Prior Shingle-only proof already showed why this is hard as a standalone route:

- Best source-verified Shingle `1656` single-car route starts at Dry Creek /
  Sweet Connie roadside parking.
- On-foot miles: 11.88
- Door-to-door p75 / p90: 260 / 292 minutes
- The official Shingle segment itself is not slow; the blocker is legal
  same-car access/return mileage.
- Exhaustive known-anchor probing did not find a strict 260-minute p90 Shingle
  solution. The best field-ready anchor remained the same Dry Creek / Sweet
  Connie roadside parking start.

## Latent Credit Found In 15A-1

I ran `field_activity_review.py` against the current exported `15A-1` audit GPX
as if it were the recorded activity, with planned segments limited to the five
Dry Creek official segments.

Command:

```bash
python3 years/2026/scripts/field_activity_review.py \
  --activity docs/field-packet/gpx/audit/15a-1-dry-creek-sweet-connie-roadside-parking-dry-creek-trail.gpx \
  --planned-outing-id 15-1 \
  --planned-segment-ids 1542,1543,1544,1545,1546 \
  --output-json years/2026/checkpoints/15a-1-latent-shingle-credit-review-2026-05-11.json
```

Result:

- Completed: 6 official segments
- Extra completed: `1656`
- Missed planned: none
- Shingle `1656`: match fraction 1.000, both endpoints 0.0000 miles from the
  track, ascent direction passed with `official_geometry_start_to_end`.
- Sheep Camp `1653`: partial only, match fraction 0.119, one endpoint not
  covered; do not count Sheep Camp from `15A-1`.

This means `15A-1` is already physically doing the Shingle climb. The field
packet is failing to account for that cross-card credit.

## Current 16A-2 Extra Repeat

I also ran the same activity-review validator against the current `16A-2` GPX.

Command:

```bash
python3 years/2026/scripts/field_activity_review.py \
  --activity docs/field-packet/gpx/audit/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail-shingle-creek-trail.gpx \
  --planned-outing-id 16-2 \
  --planned-segment-ids 1656,1653 \
  --output-json years/2026/checkpoints/16a-2-activity-review-current-route-2026-05-11.json
```

Result:

- Completed: 5 official segments
- Planned completed: `1653`, `1656`
- Extra completed: `1542`, `1543`, `1544`
- Dry Creek `1545` and `1546` are only partial.

So if `15A-1` and the current `16A-2` are both run, the plan pays major repeat
mileage in both directions: `15A-1` already covers Shingle, while `16A-2`
re-covers lower Dry Creek segments.

## Public Route Behavior

Public route sources support the same physical pattern:

- Strava exposes a public route titled `Shingle Creek and Dry Creek Loop`.
- SWIMBA's Shingle/Dry Creek route starts from Dry Creek Trailhead parking on
  Bogus Basin Road across from Sweet Connie, goes Dry Creek to Shingle, climbs
  Shingle, then descends Dry Creek.
- AllTrails describes Dry Creek - Shingle Creek Loop as a 13.6-mile loop with
  trail running use, parking at the trailhead, seasonal muddy-condition risk,
  creek crossings, minimal shade, and a 6.5-7 hour hiking estimate.
- Idaho Trails Association describes the area as a 13.8-mile loop, says parking
  is available on the side of Bogus Basin Road in two places, and frames Shingle
  as the steeper, shorter ascent option while Dry Creek is longer and gentler.

The public behavior is not "find a magic Shingle parking spot." It is
Shingle-up / Dry-down from the Bogus Basin Road pullout. That is exactly what
the current `15A-1` GPX already appears to do.

## Optimization Options

### Option A: Conservative Segment-First Execution

Keep the route cards unchanged for now, but schedule `15A-1` before `16A-2`.
After the actual BTC app recording for `15A-1`, run `field_activity_review.py`
on the real activity. If `1656` appears as `extra_completed_segment_ids`, apply
that progress and recertify the remaining menu. The active `16A-2` should then
collapse to Sheep Camp-only work.

This is the safest operational path because it lets actual activity geometry
decide credit.

### Option B: Promote A New 15A-1+Shingle Card

Regenerate the field packet so `15A-1` intentionally claims:

- Dry Creek `1542`, `1543`, `1544`, `1545`, `1546`
- Shingle `1656`

Then replace `16A-2` with a Sheep Camp-only card:

- Sheep Camp `1653`
- 3.30 on-foot miles
- 106 p75 / 119 p90

This is the better planned menu, but it needs field-packet regeneration and a
coverage audit because it crosses the current package boundary.

### Option C: Add A Durable Latent-Credit Audit

Add a generator/audit pass that reviews every exported route GPX against all
official segments and reports completed but unclaimed official segments. This
would have caught `15A-1 -> 1656` automatically.

This is the structural fix. It avoids route-specific exception logic and helps
the field-day layer preserve future-day optimization after any route physically
covers extra official credit.

## Recommendation

Use Option A immediately for field planning, then implement Option C before
rewriting public field-packet route cards. Option B is the likely final menu
shape once the latent-credit audit and field-packet regeneration can prove it
cleanly.

Do not keep optimizing `16A-2` as a standalone Shingle+Sheep route unless the
real `15A-1` activity fails to complete Shingle `1656`. The current evidence
says standalone `16A-2` is mostly redundant after `15A-1`.

## Remaining Proof

- Validate actual BTC app activity geometry after the `15A-1` field run.
- Check current Ridge to Rivers / Forest Service conditions, heat, water,
  parking capacity, and signage before using the route.
- Confirm the phone field-day layer can express a dependency: `16A-2` becomes
  Sheep Camp-only after `15A-1` validates Shingle.
- Regenerate and audit the field packet before publishing a new official route
  card.
