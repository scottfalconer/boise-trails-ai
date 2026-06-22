# Latent-Credit Delta Repricing Audit

Generated: 2026-06-22T01:56:58Z
Status: `pairwise_savings_only`

## Summary

- Routes audited: 28
- Latent route relationships: 20
- Unique latent official segments: 36
- Pairwise full-removal relationships: 2
- Pairwise partial-shrink relationships: 18
- Current-calendar removed routes: 0
- Current-calendar partial reprices needed: 7
- Current-calendar proven savings: 0.00 on-foot mi, 0 p75 min, 0 p90 min

## Current Calendar Route Removals

- None.

## Pairwise Full-Removal Opportunities

| Source route | Future/owner route | Order status | Latent ids | Saved on-foot mi | Saved p75 | Saved p90 |
|---|---|---|---:|---:|---:|---:|
| 17-1: 17 | 18-2: 18B | owner_scheduled_before_source | 1655, 1721 | 6.82 | 165 | 185 |
| 18-1: 18A | 18-2: 18B | owner_scheduled_before_source | 1655, 1721 | 6.82 | 165 | 185 |

## Partial Shrink Candidates

| Route | Date | Already credited ids | Remaining ids | Current on-foot mi | Current p75 |
|---|---|---:|---:|---:|---:|
| 13-1: 13 |  | 1558, 1685, 1748 | 1555, 1563, 1564, 1649, 1650, 1651, 1652, 1681, 1682, 1683, 1684, 1710, 1711 | 32.47 | 497 |
| 6-1: 6 |  | 1520 | 1508, 1509, 1516, 1519, 1521, 1597, 1709 | 22.41 | 409 |
| 2-1: 2 |  | 1532 | 1533, 1534, 1535, 1583, 1585, 1586, 1587, 1588, 1589, 1615, 1616, 1725, 1726, 1727, 1728, 1729, 1730, 1751 | 18.80 | 332 |
| 17-1: 17 |  | 1553, 1554, 1680, 1713, 1750 | 1488, 1489, 1490, 1491, 1492, 1493, 1540 | 16.02 | 386 |
| 18-1: 18A |  | 1679 | 1501, 1502, 1503, 1678, 1703, 1732, 1733, 1734, 1735, 1736 | 15.30 | 357 |
| 12-1: 12 |  | 1528 | 1483, 1484, 1485, 1486, 1524, 1525, 1526, 1527, 1660 | 13.44 | 243 |
| 15-2: 15B |  | 1577 | 1523, 1576 | 8.43 | 147 |

## Scope Boundary

- Full-removal rows are the only rows with proven on-foot and p75/p90 savings in this artifact.
- Partial-shrink rows are intentionally priced at zero proven savings until a generated replacement route card exists.
- This audit does not mark BTC progress; it only prices what the active route menu could stop asking the runner to do after validated latent credit is accepted.
