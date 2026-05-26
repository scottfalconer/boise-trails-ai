# Field latent credit audit

- Status: `passed`
- Routes audited: 49
- Routes needing repair: 0
- Routes with reconciled latent credit: 23
- Unexpected latent official segments: 46
- Unreconciled latent segments claimed by another active route: 0
- Reconciled latent segments claimed by another active route: 46
- Unclaimed uncompleted latent segments: 0
- Repeat-only latent completed segments: 0

## Scope

- This audit proves segment-credit provenance: latent official segments in route GPX files are either declared against another active route card, already completed at export, or surfaced as repair debt.
- A passing result makes the packet more executable and auditable; it does not prove lower total on-foot miles, lower p75/p90 time, better sequencing, or net human-effort reduction.
- Effort reduction still requires route-card replacement or field-day repricing after validated activity progress changes the remaining segment set.

## Reconciled latent credit

### 114-1: FD14A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/polecat-gulch-doe-ridge-fd14a.gpx`
- Declared owned by other active routes: 1599, 1604
- Segment details:
  - 1599 Polecat Loop; claimed by 118-1: FD18A
  - 1604 Polecat Loop; claimed by 118-1: FD18A

### 114-3: FD14C
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/polecat-gulch-quick-draw-fd14c.gpx`
- Declared owned by other active routes: 1541, 1599, 1604
- Segment details:
  - 1541 Doe Ridge; claimed by 114-1: FD14A
  - 1599 Polecat Loop; claimed by 118-1: FD18A
  - 1604 Polecat Loop; claimed by 118-1: FD18A

### 112-2: FD12B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/hillside-to-hollow-who-now-loop-fd12b.gpx`
- Declared owned by other active routes: 1755
- Segment details:
  - 1755 Buena Vista Trail; claimed by 112-1: FD12A

### 105-2: 4A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/upper-hulls-gulch-bob-s-4a.gpx`
- Declared owned by other active routes: 1577
- Segment details:
  - 1577 Highlands Trail; claimed by 122-1: FD22A

### 114-2: FD14B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/polecat-gulch-chbh-connector-fd14b.gpx`
- Declared owned by other active routes: 1541, 1599, 1604
- Segment details:
  - 1541 Doe Ridge; claimed by 114-1: FD14A
  - 1599 Polecat Loop; claimed by 118-1: FD18A
  - 1604 Polecat Loop; claimed by 118-1: FD18A

### 119-2: FD19B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/camels-back-hulls-gulch-lower-hull-s-gulch-fd19b.gpx`
- Declared owned by other active routes: 1484, 1532, 1533
- Segment details:
  - 1484 8th Street Motorcycle Trail; claimed by 123-1: FD23A
  - 1532 Crestline Trail; claimed by 122-2: FD22B
  - 1533 Crestline Trail; claimed by 122-2: FD22B

### 122-2: FD22B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/camels-back-hulls-gulch-crestline-fd22b.gpx`
- Declared owned by other active routes: 1615
- Segment details:
  - 1615 Red Cliffs; claimed by 119-2: FD19B

### 129-1: 16A-2
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/dry-creek-sheep-camp-16a-2.gpx`
- Declared owned by other active routes: 1542, 1543
- Segment details:
  - 1542 Dry Creek Trail; claimed by 128-2: 15A-1
  - 1543 Dry Creek Trail; claimed by 128-2: 15A-1

### 119-3: FD19C
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/rocky-canyon-shane-s-fd19c.gpx`
- Declared owned by other active routes: 1593, 1594, 1629, 1630, 1631, 1683, 1748
- Segment details:
  - 1593 Mountain Cove; claimed by 115-1: FD15A
  - 1594 Mountain Cove; claimed by 115-1: FD15A
  - 1629 Ridge Crest; claimed by 115-1: FD15A
  - 1630 Ridge Crest; claimed by 115-1: FD15A
  - 1631 Ridge Crest; claimed by 115-1: FD15A
  - 1683 Three Bears Trail; claimed by 120-1: FD20A
  - 1748 Two Point; claimed by 104-1: FD04A

### 123-2: FD23B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/upper-hulls-gulch-sidewinder-fd23b.gpx`
- Declared owned by other active routes: 1483, 1484, 1485, 1532
- Segment details:
  - 1483 8th Street Motorcycle Trail; claimed by 123-1: FD23A
  - 1484 8th Street Motorcycle Trail; claimed by 123-1: FD23A
  - 1485 8th Street Motorcycle Trail; claimed by 123-1: FD23A
  - 1532 Crestline Trail; claimed by 122-2: FD22B

### 105-1: FD05A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/upper-hulls-gulch-hull-s-gulch-interpretive-trail-fd05a.gpx`
- Declared owned by other active routes: 1483
- Segment details:
  - 1483 8th Street Motorcycle Trail; claimed by 123-1: FD23A

### 116-2: 15B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/hidden-springs-red-tail-15b.gpx`
- Declared owned by other active routes: 1497
- Segment details:
  - 1497 Bitterbrush Trail; claimed by 109-2: 10B

### 109-2: 10B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/hidden-springs-bitterbrush-10b.gpx`
- Declared owned by other active routes: 1520, 1521, 1619, 1622, 1623, 1624
- Segment details:
  - 1520 Chukar Butte Trail; claimed by 103-1: FD03A
  - 1521 Chukar Butte Trail; claimed by 103-1: FD03A
  - 1619 Red Tail Trail; claimed by 116-2: 15B
  - 1622 Red Tail Trail; claimed by 116-2: 15B
  - 1623 Red Tail Trail; claimed by 116-2: 15B
  - 1624 Red Tail Trail; claimed by 116-2: 15B

### 107-2: FD07B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/bogus-basin-deer-point-fd07b.gpx`
- Declared owned by other active routes: 1655
- Segment details:
  - 1655 Shindig; claimed by 131-1: 18

### 125-2: FD25B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/bogus-basin-the-face-fd25b.gpx`
- Declared owned by other active routes: 1679
- Segment details:
  - 1679 Tempest Trail; claimed by 131-1: 18

### 125-1: FD25A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/bogus-basin-elk-meadows-fd25a.gpx`
- Declared owned by other active routes: 1655
- Segment details:
  - 1655 Shindig; claimed by 131-1: 18

### 123-3: FD23C
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/dry-creek-corrals-fd23c.gpx`
- Declared owned by other active routes: 1498, 1499, 1500, 1523, 1576, 1577
- Segment details:
  - 1498 Bob's Trail; claimed by 105-2: 4A
  - 1499 Bob's Trail; claimed by 105-2: 4A
  - 1500 Bob's Trail; claimed by 105-2: 4A
  - 1523 Connector; claimed by 128-1: FD28A
  - 1576 Highlands Trail; claimed by 122-1: FD22A
  - 1577 Highlands Trail; claimed by 122-1: FD22A

### 104-1: FD04A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/military-reserve-two-point-fd04a.gpx`
- Declared owned by other active routes: 1593, 1594, 1629, 1630, 1631, 1649, 1650, 1651, 1683, 1684, 1695
- Segment details:
  - 1593 Mountain Cove; claimed by 115-1: FD15A
  - 1594 Mountain Cove; claimed by 115-1: FD15A
  - 1629 Ridge Crest; claimed by 115-1: FD15A
  - 1630 Ridge Crest; claimed by 115-1: FD15A
  - 1631 Ridge Crest; claimed by 115-1: FD15A
  - 1649 Shane's Trail; claimed by 119-3: FD19C
  - 1650 Shane's Trail; claimed by 119-3: FD19C
  - 1651 Shane's Trail; claimed by 119-3: FD19C
  - 1683 Three Bears Trail; claimed by 120-1: FD20A
  - 1684 Three Bears Trail; claimed by 120-1: FD20A
  - 1695 Watchman Trail; claimed by 111-1: 14

### 106-1: FD06A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/upper-military-reserve-fat-tire-traverse-fd06a.gpx`
- Declared owned by other active routes: 1483, 1484, 1485, 1486, 1564, 1751
- Segment details:
  - 1483 8th Street Motorcycle Trail; claimed by 123-1: FD23A
  - 1484 8th Street Motorcycle Trail; claimed by 123-1: FD23A
  - 1485 8th Street Motorcycle Trail; claimed by 123-1: FD23A
  - 1486 8th Street Motorcycle Trail; claimed by 123-1: FD23A
  - 1564 Freestone Ridge; claimed by 120-1: FD20A
  - 1751 Hull's Gulch Interpretive; claimed by 105-1: FD05A

### 111-1: 14
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/rocky-canyon-orchard-gulch-14.gpx`
- Declared owned by other active routes: 1558, 1685
- Segment details:
  - 1558 Femrite's Patrol; claimed by 104-1: FD04A
  - 1685 Three Bears Trail; claimed by 120-1: FD20A

### 115-1: FD15A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/military-reserve-military-reserve-connection-fd15a.gpx`
- Declared owned by other active routes: 1748
- Segment details:
  - 1748 Two Point; claimed by 104-1: FD04A

### 120-1: FD20A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/rocky-canyon-three-bears-fd20a.gpx`
- Declared owned by other active routes: 1593, 1594, 1629
- Segment details:
  - 1593 Mountain Cove; claimed by 115-1: FD15A
  - 1594 Mountain Cove; claimed by 115-1: FD15A
  - 1629 Ridge Crest; claimed by 115-1: FD15A

### 131-1: 18
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/bogus-basin-brewers-byway-ext-18.gpx`
- Declared owned by other active routes: 1553, 1750
- Segment details:
  - 1553 Elk Meadows Trail; claimed by 125-1: FD25A
  - 1750 Around the Mountain Trail (ascent); claimed by 126-1: FD26A
