# 10A-MS-08 Access Verification - 2026-05-10

## Question

Can `10A-MS-08` be field-certified as a legal same-day re-park route using:

- `North Burnt Car Place road-parking anchor`
- `Harlow's / Hidden Springs west access probe`

## Result

No. Keep `10A-MS-08` parking-gated and do not promote it into the field packet.

The route remains useful as a design target because it saves 3.38 on-foot miles
and 43 p75 minutes on paper, but the access evidence does not certify both
starts as legal, practical, repeatable car starts.

## Evidence Checked

External sources:

- Avimor Trails and Outdoors page, checked 2026-05-10:
  `https://www.avimor.com/trails-and-outdoors`
- Avimor 2024 East Side Recreation Trails Map, checked 2026-05-10:
  `https://s3.amazonaws.com/buildercloud/18d76d01b5394cb3ac4ce724a65248e3.pdf`
- MTB Project Harlow Hollows page, checked 2026-05-10:
  `https://www.mtbproject.com/trail/7002138/harlow-hollows`

Local/generated evidence:

- `years/2026/checkpoints/multi-start-alternative-audit-2026-05-08.json`
- `years/2026/checkpoints/route-pain-index-2026-05-10.json`
- `docs/field-packet/field-tool-data.json`
- Google Earth Pro imagery check, 2026-05-10.
- Prior local Google Maps Street View check:
  `years/2026/outputs/private/multi-start-alternatives/parking-earth-screenshots/current-assumed-paved-road-anchors/streetview-review-2026-05-08.md`

## Access Findings

### North Burnt Car Place Road-Parking Anchor

Status: `parking_gated`

Supporting evidence:

- The generated multi-start audit identifies this anchor as an OSM public-road
  probe on a residential asphalt road within the allowed 0.10-mile access
  threshold.
- Google Earth imagery shows a residential street at the anchor and a nearby
  trail/drainage connection toward the Harlow/Burnt Car area.
- The local Street View review found no Street View imagery at the exact anchor,
  so it cannot resolve curb signage, no-parking signs, HOA rules, private-road
  signs, or shoulder safety.

Blocking evidence:

- This is not shown as a formal trailhead lot in the checked Avimor map/source
  evidence.
- The Avimor source frames the trail system around Avimor resident use and
  permission workflows for some access/parking uses, so a generic residential
  road probe is not enough to certify legal public parking.
- No current source or imagery checked here confirms that a non-resident can
  park at this residential anchor for trail access.

Decision:

Physically plausible, but not field-certifiable without on-site signage review
or explicit Avimor/land-manager confirmation.

### Harlow's / Hidden Springs West Access Probe

Status: `not_a_certified_car_start`

Supporting evidence:

- The current active `10A` field packet uses this point as the parking/start
  label, but the multi-start audit still marks it `manual_required` and
  `field_ready: false`.
- Google Earth Pro places the probe on a drainage/trail corridor, not on a
  visible vehicle parking area or formal trailhead lot.
- The Avimor trail map confirms Harlow's Hollows and Harlow's Hollows Connector
  exist in this area, but it does not show this probe as a parking node.
- MTB Project notes that Harlow Hollows lower-route orientation has been
  disrupted by house-construction grading, increasing cue and field-confirmation
  risk around the neighborhood edge.

Blocking evidence:

- No visible car-access/parking feature at the probe point from Google Earth.
- No Street View/source confirmation that this is a legal parking or re-park
  point.
- The route card could potentially describe this as a trail/access waypoint, but
  not as a car-to-car start until a real parked start is substituted and cued.

Decision:

Do not use this as the second parked start for `10A-MS-08`.

## Certification Decision

`10A-MS-08` is not certifiable as a legal same-day re-park route today.

To promote a `10A` replacement, one of these must happen:

1. Avimor or field signage confirms both residential starts are legal,
   repeatable, and acceptable for trail access, including non-resident parking
   if that matters for the user's challenge use.
2. The second component is redesigned from a known parking node, such as an
   Avimor-designated parking/trailhead point, with added connector mileage and
   p75/p90 timing recalculated.
3. The existing `10A` route remains active until a substitute passes the full
   field-packet certification chain.

## Field-Guide Impact

- Do not promote `10A-MS-08` into `2026-field-menu-replacements`.
- Keep `10A` in the route-pain index as an access-verification target, but mark
  the current `10A-MS-08` access result as blocked/pending redesign.
- If a future field check accepts the North Burnt Car Place anchor, the Harlow
  west probe still needs a separate certified parked-start replacement before
  this exact same-day re-park plan can be used.
