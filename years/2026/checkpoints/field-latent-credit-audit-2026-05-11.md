# Field latent credit audit

- Status: `passed`
- Routes audited: 43
- Routes needing repair: 0
- Routes with reconciled latent credit: 17
- Unexpected latent official segments: 35
- Unreconciled latent segments claimed by another active route: 0
- Reconciled latent segments claimed by another active route: 35
- Unclaimed uncompleted latent segments: 0
- Repeat-only latent completed segments: 0

## Scope

- This audit proves segment-credit provenance: latent official segments in route GPX files are either declared against another active route card, already completed at export, or surfaced as repair debt.
- A passing result makes the packet more executable and auditable; it does not prove lower total on-foot miles, lower p75/p90 time, better sequencing, or net human-effort reduction.
- Effort reduction still requires route-card replacement or field-day repricing after validated activity progress changes the remaining segment set.

## Reconciled latent credit

### 114-1: FD14A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/fd14a-cartwright-doe-ridge.gpx`
- Declared owned by other active routes: 1599, 1604
- Segment details:
  - 1599 Polecat Loop; claimed by 118-1: FD18A
  - 1604 Polecat Loop; claimed by 118-1: FD18A

### 105-2: 4A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/4a-bob-s-bob-s-trail-urban-connector.gpx`
- Declared owned by other active routes: 1577
- Segment details:
  - 1577 Highlands Trail; claimed by 123-1: 12

### 114-2: FD14B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/fd14b-cartwright-chbh-connector-quick-draw.gpx`
- Declared owned by other active routes: 1541, 1599, 1604
- Segment details:
  - 1541 Doe Ridge; claimed by 114-1: FD14A
  - 1599 Polecat Loop; claimed by 118-1: FD18A
  - 1604 Polecat Loop; claimed by 118-1: FD18A

### 119-2: FD19B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/fd19b-hulls-gulch-lower-hull-s-gulch-trail-red-cliffs.gpx`
- Declared owned by other active routes: 1484, 1532, 1533
- Segment details:
  - 1484 8th Street Motorcycle Trail; claimed by 123-1: 12
  - 1532 Crestline Trail; claimed by 122-1: FD22B
  - 1533 Crestline Trail; claimed by 122-1: FD22B

### 122-1: FD22B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/fd22b-hulls-gulch-crestline-trail.gpx`
- Declared owned by other active routes: 1615
- Segment details:
  - 1615 Red Cliffs; claimed by 119-2: FD19B

### 129-1: 16A-2
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail.gpx`
- Declared owned by other active routes: 1542, 1543
- Segment details:
  - 1542 Dry Creek Trail; claimed by 128-2: 15A-1
  - 1543 Dry Creek Trail; claimed by 128-2: 15A-1

### 105-1: FD05A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/fd05a-8th-street-atv-parking-area-hull-s-gulch-interpretive.gpx`
- Declared owned by other active routes: 1483
- Segment details:
  - 1483 8th Street Motorcycle Trail; claimed by 123-1: 12

### 109-2: 10B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/10b-dry-creek-parking-area-trailhead-bitterbrush-trail-currant-creek.gpx`
- Declared owned by other active routes: 1520, 1521, 1619, 1624
- Segment details:
  - 1520 Chukar Butte Trail; claimed by 103-1: FD03A
  - 1521 Chukar Butte Trail; claimed by 103-1: FD03A
  - 1619 Red Tail Trail; claimed by 116-2: 15B
  - 1624 Red Tail Trail; claimed by 116-2: 15B

### 125-2: FD25B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/fd25b-pioneer-lodge-parking-area-the-face-trail.gpx`
- Declared owned by other active routes: 1679
- Segment details:
  - 1679 Tempest Trail; claimed by 131-1: 18

### 125-1: FD25A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/fd25a-simplot-lodge-parking-area-elk-meadows-trail.gpx`
- Declared owned by other active routes: 1655
- Segment details:
  - 1655 Shindig; claimed by 131-1: 18

### 104-1: FD04A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/fd04a-freestone-creek-two-point-shane-s-connector-femrite-s-patrol-shane-s-trail.gpx`
- Declared owned by other active routes: 1593, 1594, 1629, 1630, 1631, 1683, 1684, 1695
- Segment details:
  - 1593 Mountain Cove; claimed by 115-1: 3
  - 1594 Mountain Cove; claimed by 115-1: 3
  - 1629 Ridge Crest; claimed by 115-1: 3
  - 1630 Ridge Crest; claimed by 115-1: 3
  - 1631 Ridge Crest; claimed by 115-1: 3
  - 1683 Three Bears Trail; claimed by 120-1: FD20A
  - 1684 Three Bears Trail; claimed by 120-1: FD20A
  - 1695 Watchman Trail; claimed by 111-1: 14

### 106-1: FD06A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/fd06a-lower-interpretive-fat-tire-traverse-curlew-connection.gpx`
- Declared owned by other active routes: 1483, 1484, 1485, 1486, 1564, 1751
- Segment details:
  - 1483 8th Street Motorcycle Trail; claimed by 123-1: 12
  - 1484 8th Street Motorcycle Trail; claimed by 123-1: 12
  - 1485 8th Street Motorcycle Trail; claimed by 123-1: 12
  - 1486 8th Street Motorcycle Trail; claimed by 123-1: 12
  - 1564 Freestone Ridge; claimed by 120-1: FD20A
  - 1751 Hull's Gulch Interpretive; claimed by 105-1: FD05A

### 111-1: 14
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/14-orchard-gulch-orchard-gulch-trail-five-mile-gulch-trail-watchman-trail.gpx`
- Declared owned by other active routes: 1558, 1685
- Segment details:
  - 1558 Femrite's Patrol; claimed by 104-1: FD04A
  - 1685 Three Bears Trail; claimed by 120-1: FD20A

### 115-1: 3
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/3-freestone-creek-military-reserve-connection-mountain-cove-central-ridge-trail-central-ri.gpx`
- Declared owned by other active routes: 1748
- Segment details:
  - 1748 Two Point; claimed by 104-1: FD04A

### 120-1: FD20A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/fd20a-freestone-creek-three-bears-trail-freestone-ridge.gpx`
- Declared owned by other active routes: 1593, 1594, 1629
- Segment details:
  - 1593 Mountain Cove; claimed by 115-1: 3
  - 1594 Mountain Cove; claimed by 115-1: 3
  - 1629 Ridge Crest; claimed by 115-1: 3

### 123-1: 12
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/12-8th-street-atv-parking-area-8th-street-motorcycle-trail-sidewinder-trail-corrals-trail-.gpx`
- Declared owned by other active routes: 1498, 1532
- Segment details:
  - 1498 Bob's Trail; claimed by 105-2: 4A
  - 1532 Crestline Trail; claimed by 122-1: FD22B

### 131-1: 18
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/18-pioneer-lodge-parking-area-brewer-s-byway-extension-brewers-byway-shindig-tempest-trail.gpx`
- Declared owned by other active routes: 1553, 1750
- Segment details:
  - 1553 Elk Meadows Trail; claimed by 125-1: FD25A
  - 1750 Around the Mountain Trail (ascent); claimed by 126-1: FD26A
