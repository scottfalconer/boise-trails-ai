# Field latent credit audit

- Status: `passed`
- Routes audited: 31
- Routes needing repair: 0
- Routes with reconciled latent credit: 16
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

### 16-2: 16A-2
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/dry-creek-sheep-camp-16a-2.gpx`
- Declared owned by other active routes: 1542, 1543
- Segment details:
  - 1542 Dry Creek Trail; claimed by 15-2: 15B
  - 1543 Dry Creek Trail; claimed by 15-2: 15B

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

### 1-3: 1B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/hillside-to-hollow-who-now-loop-1b.gpx`
- Declared owned by other active routes: 1755
- Segment details:
  - 1755 Buena Vista Trail; claimed by 1-2: 1A-2

### 10-1: 10B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/hidden-springs-bitterbrush-10b.gpx`
- Declared owned by other active routes: 1520, 1619, 1622, 1623, 1624
- Segment details:
  - 1520 Chukar Butte Trail; claimed by 6-1: 6
  - 1619 Red Tail Trail; claimed by 15-1: 15A
  - 1622 Red Tail Trail; claimed by 15-1: 15A
  - 1623 Red Tail Trail; claimed by 15-1: 15A
  - 1624 Red Tail Trail; claimed by 15-1: 15A

### 18-2: 18B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/bogus-basin-shindig-18b.gpx`
- Declared owned by other active routes: 1553, 1554, 1679, 1680
- Segment details:
  - 1553 Elk Meadows Trail; claimed by 17-1: 17
  - 1554 Elk Meadows Trail; claimed by 17-1: 17
  - 1679 Tempest Trail; claimed by 18-1: 18A
  - 1680 The Face Trail; claimed by 17-1: 17

### 3-1: 3
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/military-reserve-military-reserve-connection-3.gpx`
- Declared owned by other active routes: 1748
- Segment details:
  - 1748 Two Point; claimed by 13-1: 13

### 12-1: 12
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/dry-creek-corrals-12.gpx`
- Declared owned by other active routes: 1498, 1499, 1500, 1532, 1576, 1577
- Segment details:
  - 1498 Bob's Trail; claimed by 4-1: 4A
  - 1499 Bob's Trail; claimed by 4-1: 4A
  - 1500 Bob's Trail; claimed by 4-1: 4A
  - 1532 Crestline Trail; claimed by 2-1: 2
  - 1576 Highlands Trail; claimed by 15-2: 15B
  - 1577 Highlands Trail; claimed by 15-2: 15B

### 14-1: 14
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/rocky-canyon-orchard-gulch-14.gpx`
- Declared owned by other active routes: 1558, 1685
- Segment details:
  - 1558 Femrite's Patrol; claimed by 13-1: 13
  - 1685 Three Bears Trail; claimed by 13-1: 13

### 16-4: 16C-2
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/dry-creek-shingle-creek-16c-2.gpx`
- Declared owned by other active routes: 1542, 1543, 1544
- Segment details:
  - 1542 Dry Creek Trail; claimed by 15-2: 15B
  - 1543 Dry Creek Trail; claimed by 15-2: 15B
  - 1544 Dry Creek Trail; claimed by 15-2: 15B

### 2-1: 2
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/camels-back-hulls-gulch-lower-hull-s-gulch-2.gpx`
- Declared owned by other active routes: 1483, 1484
- Segment details:
  - 1483 8th Street Motorcycle Trail; claimed by 12-1: 12
  - 1484 8th Street Motorcycle Trail; claimed by 12-1: 12

### 18-1: 18A
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/bogus-basin-brewers-byway-ext-18a.gpx`
- Declared owned by other active routes: 1553, 1655, 1713, 1750
- Segment details:
  - 1553 Elk Meadows Trail; claimed by 17-1: 17
  - 1655 Shindig; claimed by 18-2: 18B
  - 1713 Sunshine XC (ascent); claimed by 17-1: 17
  - 1750 Around the Mountain Trail (ascent); claimed by 17-1: 17

### 15-2: 15B
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/dry-creek-highlands-15b.gpx`
- Declared owned by other active routes: 1527, 1528
- Segment details:
  - 1527 Corrals Trail; claimed by 12-1: 12
  - 1528 Corrals Trail; claimed by 12-1: 12

### 17-1: 17
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/bogus-basin-sunshine-xc-17.gpx`
- Declared owned by other active routes: 1655, 1679, 1721
- Segment details:
  - 1655 Shindig; claimed by 18-2: 18B
  - 1679 Tempest Trail; claimed by 18-1: 18A
  - 1721 Lodge Trail; claimed by 18-2: 18B

### 6-1: 6
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/dry-creek-peggy-s-6.gpx`
- Declared owned by other active routes: 1541, 1599, 1604, 1610, 1666
- Segment details:
  - 1541 Doe Ridge; claimed by 5-2: 5B
  - 1599 Polecat Loop; claimed by 5-2: 5B
  - 1604 Polecat Loop; claimed by 5-2: 5B
  - 1610 Quick Draw; claimed by 5-2: 5B
  - 1666 Sweet Connie Trail (ascent); claimed by 16-1: 16A-1

### 13-1: 13
- GPX: `/Users/scott/dev/boise-trails-ai/docs/field-packet/gpx/official/rocky-canyon-three-bears-13.gpx`
- Declared owned by other active routes: 1593, 1594, 1629, 1630, 1631, 1695
- Segment details:
  - 1593 Mountain Cove; claimed by 3-1: 3
  - 1594 Mountain Cove; claimed by 3-1: 3
  - 1629 Ridge Crest; claimed by 3-1: 3
  - 1630 Ridge Crest; claimed by 3-1: 3
  - 1631 Ridge Crest; claimed by 3-1: 3
  - 1695 Watchman Trail; claimed by 14-1: 14
