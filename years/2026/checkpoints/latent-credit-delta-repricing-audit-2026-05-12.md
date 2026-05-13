# Latent-Credit Delta Repricing Audit

Generated: 2026-05-13T03:45:24Z
Status: `pairwise_savings_only`

## Summary

- Routes audited: 44
- Latent route relationships: 33
- Unique latent official segments: 38
- Pairwise full-removal relationships: 2
- Pairwise partial-shrink relationships: 31
- Current-calendar removed routes: 0
- Current-calendar partial reprices needed: 10
- Current-calendar proven savings: 0.00 on-foot mi, 0 p75 min, 0 p90 min

## Current Calendar Route Removals

- None.

## Pairwise Full-Removal Opportunities

| Source route | Future/owner route | Order status | Latent ids | Saved on-foot mi | Saved p75 | Saved p90 |
|---|---|---|---:|---:|---:|---:|
| 104-1: FD04A | 119-3: FD19C | owner_scheduled_before_source | 1649, 1650, 1651 | 4.76 | 109 | 123 |
| 114-2: FD14B | 114-1: FD14A | owner_scheduled_before_source | 1541 | 1.08 | 58 | 65 |

## Partial Shrink Candidates

| Route | Date | Already credited ids | Remaining ids | Current on-foot mi | Current p75 |
|---|---|---:|---:|---:|---:|
| 118-1: FD18A | 2026-07-14 | 1599, 1604 | 1597, 1598, 1600, 1601, 1602, 1603 | 13.32 | 254 |
| 120-1: FD20A | 2026-07-15 | 1564, 1683, 1684, 1685 | 1563, 1681, 1682 | 13.10 | 255 |
| 123-1: 12 | 2026-06-20 | 1484 | 1483, 1485, 1486, 1524, 1525, 1526, 1527, 1528, 1576, 1577, 1660 | 12.86 | 262 |
| 115-1: 3 | 2026-07-09 | 1593, 1594, 1629, 1630, 1631 | 1510, 1511, 1512, 1513, 1514, 1515, 1522, 1529, 1530, 1531, 1548, 1549, 1550, 1551, 1552, 1574, 1575, 1590, 1591, 1592, 1627, 1628, 1720 | 12.13 | 250 |
| 131-1: 18 | 2026-07-18 | 1655, 1679 | 1501, 1502, 1503, 1678, 1703, 1721, 1732, 1733, 1734, 1735, 1736 | 11.25 | 320 |
| 111-1: 14 | 2026-07-03 | 1695 | 1560, 1561, 1562, 1595, 1694 | 10.74 | 242 |
| 104-1: FD04A | 2026-06-24 | 1748 | 1558, 1652 | 9.55 | 204 |
| 116-2: 15B | 2026-07-10 | 1619, 1624 | 1584, 1618, 1620, 1621, 1622, 1623 | 4.87 | 148 |
| 122-1: FD22B | 2026-07-17 | 1532, 1533 | 1534, 1535 | 4.46 | 104 |
| 105-2: 4A | 2026-06-25 | 1498 | 1499, 1500, 1690 | 4.07 | 97 |

## Scope Boundary

- Full-removal rows are the only rows with proven on-foot and p75/p90 savings in this artifact.
- Partial-shrink rows are intentionally priced at zero proven savings until a generated replacement route card exists.
- This audit does not mark BTC progress; it only prices what the active route menu could stop asking the runner to do after validated latent credit is accepted.
