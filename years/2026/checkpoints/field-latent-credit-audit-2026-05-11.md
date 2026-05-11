# Field latent credit audit

- Status: `passed`
- Routes audited: 31
- Routes needing repair: 0
- Routes with reconciled latent credit: 15
- Unexpected latent official segments: 42
- Unreconciled latent segments claimed by another active route: 0
- Reconciled latent segments claimed by another active route: 42
- Unclaimed uncompleted latent segments: 0
- Repeat-only latent completed segments: 0

## Scope

- This audit proves segment-credit provenance: latent official segments in route GPX files are either declared against another active route card, already completed at export, or surfaced as repair debt.
- A passing result makes the packet more executable and auditable; it does not prove lower total on-foot miles, lower p75/p90 time, better sequencing, or net human-effort reduction.
- Effort reduction still requires route-card replacement or field-day repricing after validated activity progress changes the remaining segment set.

## Reconciled latent credit

### 4-3: 4A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/4a-bob-s-bob-s-trail-urban-connector.gpx`
- Declared owned by other active routes: 1577
- Segment details:
  - 1577 Highlands Trail; claimed by 15-2: 15A-2

### 4-1: 4C-1
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/4c-1-warm-springs-golf-course-tram-trail-table-rock-quarry-trail.gpx`
- Declared owned by other active routes: 1633, 1634, 1635, 1636, 1638, 1640, 1641, 1676
- Segment details:
  - 1633 Rock Garden; claimed by 4-2: 4C-2
  - 1634 Rock Garden; claimed by 4-2: 4C-2
  - 1635 Rock Island; claimed by 4-2: 4C-2
  - 1636 Rock Island; claimed by 4-2: 4C-2
  - 1638 Rock Island; claimed by 4-2: 4C-2
  - 1640 Rock Island; claimed by 4-2: 4C-2
  - 1641 Rock Island; claimed by 4-2: 4C-2
  - 1676 Table Rock Trail; claimed by 4-2: 4C-2

### 16-2: 16A-2
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail.gpx`
- Declared owned by other active routes: 1542, 1543
- Segment details:
  - 1542 Dry Creek Trail; claimed by 15-1: 15A-1
  - 1543 Dry Creek Trail; claimed by 15-1: 15A-1

### 15-2: 15A-2
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/15a-2-bob-s-highlands-trail-connector.gpx`
- Declared owned by other active routes: 1528
- Segment details:
  - 1528 Corrals Trail; claimed by 12-1: 12

### 8-1: 8A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/8a-homestead-harris-ridge-trail.gpx`
- Declared owned by other active routes: 1722
- Segment details:
  - 1722 Peace Valley Overlook; claimed by 8-2: 8B

### 1-3: 1B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/1b-harrison-hollow-who-now-loop-trail-harrison-ridge-harrison-hollow-kemper-s-ridge-trail-.gpx`
- Declared owned by other active routes: 1755
- Segment details:
  - 1755 Buena Vista Trail; claimed by 1-2: 1A-2

### 10-2: 10B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/10b-dry-creek-parking-area-trailhead-bitterbrush-trail-currant-creek.gpx`
- Declared owned by other active routes: 1520, 1521, 1619, 1624
- Segment details:
  - 1520 Chukar Butte Trail; claimed by 6-1: 6
  - 1521 Chukar Butte Trail; claimed by 6-1: 6
  - 1619 Red Tail Trail; claimed by 15-3: 15B
  - 1624 Red Tail Trail; claimed by 15-3: 15B

### 14-1: 14
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/14-orchard-gulch-orchard-gulch-trail-five-mile-gulch-trail-watchman-trail.gpx`
- Declared owned by other active routes: 1558, 1685
- Segment details:
  - 1558 Femrite's Patrol; claimed by 13-1: 13
  - 1685 Three Bears Trail; claimed by 13-1: 13

### 3-1: 3
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/3-freestone-creek-military-reserve-connection-mountain-cove-central-ridge-trail-central-ri.gpx`
- Declared owned by other active routes: 1748
- Segment details:
  - 1748 Two Point; claimed by 13-1: 13

### 12-1: 12
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/12-8th-street-atv-parking-area-8th-street-motorcycle-trail-sidewinder-trail-corrals-trail.gpx`
- Declared owned by other active routes: 1498, 1532, 1576, 1577
- Segment details:
  - 1498 Bob's Trail; claimed by 4-3: 4A
  - 1532 Crestline Trail; claimed by 2-1: 2
  - 1576 Highlands Trail; claimed by 15-2: 15A-2
  - 1577 Highlands Trail; claimed by 15-2: 15A-2

### 18-1: 18
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/18-pioneer-lodge-parking-area-brewer-s-byway-extension-brewers-byway-shindig-tempest-trail.gpx`
- Declared owned by other active routes: 1553, 1750
- Segment details:
  - 1553 Elk Meadows Trail; claimed by 17-1: 17
  - 1750 Around the Mountain Trail (ascent); claimed by 17-1: 17

### 2-1: 2
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/2-hulls-gulch-lower-hull-s-gulch-trail-hull-s-gulch-interpretive-crestline-trail-red-cliff.gpx`
- Declared owned by other active routes: 1483, 1484
- Segment details:
  - 1483 8th Street Motorcycle Trail; claimed by 12-1: 12
  - 1484 8th Street Motorcycle Trail; claimed by 12-1: 12

### 17-1: 17
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/17-simplot-lodge-parking-area-sunshine-xc-deer-point-trail-around-the-mountain-trail-the-f.gpx`
- Declared owned by other active routes: 1655, 1679, 1721
- Segment details:
  - 1655 Shindig; claimed by 18-1: 18
  - 1679 Tempest Trail; claimed by 18-1: 18
  - 1721 Lodge Trail; claimed by 18-1: 18

### 6-1: 6
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/6-cartwright-peggy-s-trail-chukar-butte-trail-cartwright-connector-cartwright-ridge-chbh-c.gpx`
- Declared owned by other active routes: 1541, 1599, 1604, 1610, 1666
- Segment details:
  - 1541 Doe Ridge; claimed by 5-2: 5B
  - 1599 Polecat Loop; claimed by 5-2: 5B
  - 1604 Polecat Loop; claimed by 5-2: 5B
  - 1610 Quick Draw; claimed by 5-2: 5B
  - 1666 Sweet Connie Trail (ascent); claimed by 16-1: 16A-1

### 13-1: 13
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/13-freestone-creek-three-bears-trail-femrite-s-patrol-freestone-ridge-two-point-shane-s-tr.gpx`
- Declared owned by other active routes: 1593, 1594, 1629, 1630, 1631, 1695
- Segment details:
  - 1593 Mountain Cove; claimed by 3-1: 3
  - 1594 Mountain Cove; claimed by 3-1: 3
  - 1629 Ridge Crest; claimed by 3-1: 3
  - 1630 Ridge Crest; claimed by 3-1: 3
  - 1631 Ridge Crest; claimed by 3-1: 3
  - 1695 Watchman Trail; claimed by 14-1: 14
