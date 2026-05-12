# Latent-Credit Delta Repricing Audit

Generated: 2026-05-12T14:00:02Z
Status: `proved_current_calendar_savings`

## Summary

- Routes audited: 50
- Latent route relationships: 45
- Unique latent official segments: 47
- Pairwise full-removal relationships: 8
- Pairwise partial-shrink relationships: 37
- Current-calendar removed routes: 2
- Current-calendar partial reprices needed: 11
- Current-calendar proven savings: 4.39 on-foot mi, 147 p75 min, 166 p90 min

## Current Calendar Route Removals

| Route | Date | Latent ids | Saved on-foot mi | Saved p75 | Saved p90 |
|---|---|---:|---:|---:|---:|
| 114-3: FD14C | 2026-07-08 | 1610 | 1.63 | 68 | 77 |
| 122-1: FD22A | 2026-07-17 | 1576, 1577 | 2.76 | 79 | 89 |

## Pairwise Full-Removal Opportunities

| Source route | Future/owner route | Order status | Latent ids | Saved on-foot mi | Saved p75 | Saved p90 |
|---|---|---|---:|---:|---:|---:|
| 104-1: FD04A | 119-3: FD19C | owner_scheduled_before_source | 1649, 1650, 1651 | 4.76 | 109 | 123 |
| 123-1: 12 | 122-1: FD22A | owner_is_future_route | 1576, 1577 | 2.76 | 79 | 89 |
| 130-1: FD30A | 127-3: FD27C | owner_scheduled_before_source | 1696 | 2.01 | 118 | 133 |
| 114-2: FD14B | 114-3: FD14C | owner_is_future_route | 1610 | 1.63 | 68 | 77 |
| 127-2: FD27B | 127-1: FD27A | owner_scheduled_before_source | 1661 | 1.49 | 104 | 117 |
| 130-1: FD30A | 127-1: FD27A | owner_scheduled_before_source | 1661 | 1.49 | 104 | 117 |
| 114-2: FD14B | 114-1: FD14A | owner_scheduled_before_source | 1541 | 1.08 | 58 | 65 |
| 114-3: FD14C | 114-1: FD14A | owner_scheduled_before_source | 1541 | 1.08 | 58 | 65 |

## Partial Shrink Candidates

| Route | Date | Already credited ids | Remaining ids | Current on-foot mi | Current p75 |
|---|---|---:|---:|---:|---:|
| 130-1: FD30A | 2026-07-12 | 1687, 1688, 1689, 1708 | 1626, 1657 | 13.62 | 315 |
| 118-1: FD18A | 2026-07-14 | 1599, 1604 | 1597, 1598, 1600, 1601, 1602, 1603 | 13.32 | 254 |
| 120-1: FD20A | 2026-07-15 | 1564, 1683, 1684, 1685 | 1563, 1681, 1682 | 13.10 | 255 |
| 123-1: 12 | 2026-06-20 | 1484 | 1483, 1485, 1486, 1524, 1525, 1526, 1527, 1528, 1660 | 12.86 | 262 |
| 115-1: 3 | 2026-07-09 | 1593, 1594, 1629, 1630, 1631 | 1510, 1511, 1512, 1513, 1514, 1515, 1522, 1529, 1530, 1531, 1548, 1549, 1550, 1551, 1552, 1574, 1575, 1590, 1591, 1592, 1627, 1628, 1720 | 12.13 | 250 |
| 131-1: 18 | 2026-07-18 | 1655, 1679 | 1501, 1502, 1503, 1678, 1703, 1721, 1732, 1733, 1734, 1735, 1736 | 11.25 | 320 |
| 111-1: 14 | 2026-07-03 | 1695 | 1560, 1561, 1562, 1595, 1694 | 10.74 | 242 |
| 104-1: FD04A | 2026-06-24 | 1748 | 1558, 1652 | 9.55 | 204 |
| 116-2: 15B | 2026-07-10 | 1619, 1624 | 1584, 1618, 1620, 1621, 1622, 1623 | 4.87 | 148 |
| 122-2: FD22B | 2026-07-17 | 1532, 1533 | 1534, 1535 | 4.46 | 104 |
| 105-2: 4A | 2026-06-25 | 1498 | 1499, 1500, 1690 | 4.07 | 97 |

## Scope Boundary

- Full-removal rows are the only rows with proven on-foot and p75/p90 savings in this artifact.
- Partial-shrink rows are intentionally priced at zero proven savings until a generated replacement route card exists.
- This audit does not mark BTC progress; it only prices what the active route menu could stop asking the runner to do after validated latent credit is accepted.
