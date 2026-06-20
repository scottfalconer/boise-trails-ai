# Field latent credit audit

- Status: `passed`
- Routes audited: 19
- Routes needing repair: 0
- Routes with reconciled latent credit: 6
- Unexpected latent official segments: 14
- Unreconciled latent segments claimed by another active route: 0
- Reconciled latent segments claimed by another active route: 14
- Unclaimed uncompleted latent segments: 0
- Repeat-only latent completed segments: 1

## Scope

- This audit proves segment-credit provenance: latent official segments in route GPX files are either declared against another active route card, already completed at export, or surfaced as repair debt.
- A passing result makes the packet more executable and auditable; it does not prove lower total on-foot miles, lower p75/p90 time, better sequencing, or net human-effort reduction.
- Effort reduction still requires route-card replacement or field-day repricing after validated activity progress changes the remaining segment set.

## Reconciled latent credit

### 4-1: 4A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/upper-hulls-gulch-bob-s-4a.gpx`
- Declared owned by other active routes: 1577
- Segment details:
  - 1577 Highlands Trail; claimed by 15-2: 15B

### 8-1: 8A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/boise-river-wma-harris-ridge-trail-8a.gpx`
- Declared owned by other active routes: 1722
- Segment details:
  - 1722 Peace Valley Overlook; claimed by 8-2: 8B

### 10-2: 10B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/hidden-springs-bitterbrush-10b.gpx`
- Declared owned by other active routes: 1520, 1619, 1622, 1623, 1624
- Segment details:
  - 1520 Chukar Butte Trail; claimed by 6-1: 6
  - 1619 Red Tail Trail; claimed by 15-1: 15A
  - 1622 Red Tail Trail; claimed by 15-1: 15A
  - 1623 Red Tail Trail; claimed by 15-1: 15A
  - 1624 Red Tail Trail; claimed by 15-1: 15A

### 14-1: 14
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/rocky-canyon-orchard-gulch-14.gpx`
- Declared owned by other active routes: 1558, 1685
- Segment details:
  - 1558 Femrite's Patrol; claimed by 13-1: 13
  - 1685 Three Bears Trail; claimed by 13-1: 13

### 2-1: 2
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/camels-back-hulls-gulch-lower-hull-s-gulch-2.gpx`
- Declared owned by other active routes: 1483, 1484
- Segment details:
  - 1483 8th Street Motorcycle Trail; claimed by 12-1: 12
  - 1484 8th Street Motorcycle Trail; claimed by 12-1: 12

### 17-1: 17
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/bogus-basin-sunshine-xc-17.gpx`
- Declared owned by other active routes: 1655, 1679, 1721
- Segment details:
  - 1655 Shindig; claimed by 18-2: 18B
  - 1679 Tempest Trail; claimed by 18-1: 18A
  - 1721 Lodge Trail; claimed by 18-2: 18B

## Repeat-only latent credit

- 1-2: 1A-2: 1755
