# Ownership Reassignment Optimization Audit

Generated: 2026-06-22T01:56:58Z
Status: `ownership_reassignment_reduces_existing_loop_work`

## Summary

- Routes audited: 28
- Ownership graph edges: 13
- Relevant components: 4
- Exact optimized relevant components: 4
- Reassigned official segments: 13 (7.99 mi)
- Order-free route cards: 15 -> 14 (1 removed)
- Order-free on-foot miles: 197.01 -> 190.19 (6.82 mi saved)
- Order-free p75/p90 saved: 165 / 185 min
- Current-calendar skip-ready savings: 0.00 mi, 0 p75 min, 0 p90 min
- Partial shrink routes needing regenerated cards: 6

## Components With Order-Free Savings

| Component | Routes | Reassigned ids | Removed routes | Saved on-foot mi | Saved p75 | Calendar note |
|---|---:|---:|---|---:|---:|---|
| C04 | 3 | 6 | 18-2: 18B | 6.82 | 165 | 1 need reorder |

## Partial Shrink Credit Moves

| Route | Lost credit ids | Retained credit ids | Replacement owner(s) | Current on-foot mi |
|---|---|---|---|---:|
| 6-1: 6 | 1520 | 1508, 1509, 1516, 1519, 1521, 1597, 1709 | 10-2: 10B | 22.41 |
| 12-1: 12 | 1528 | 1483, 1484, 1485, 1486, 1524, 1525, 1526, 1527, 1532, 1660 | 15-2: 15B | 13.44 |
| 15-2: 15B | 1577 | 1523, 1528, 1576 | 4-1: 4A | 8.43 |
| 2-1: 2 | 1532 | 1533, 1534, 1535, 1583, 1585, 1586, 1587, 1588, 1589, 1615, 1616, 1725, 1726, 1727, 1728, 1729, 1730, 1751 | 12-1: 12 | 18.80 |
| 13-1: 13 | 1558, 1685, 1748 | 1555, 1563, 1564, 1649, 1650, 1651, 1652, 1681, 1682, 1683, 1684, 1710, 1711 | 14-1: 14, 3-1: 3 | 32.47 |
| 17-1: 17 | 1553, 1680, 1713, 1750 | 1488, 1489, 1490, 1491, 1492, 1493, 1540, 1554 | 18-1: 18A | 16.02 |

## Scope Boundary

- This audit separates physical traversal from credit ownership; a route can physically stay unchanged while some official segment credit moves to a different route card.
- Order-free savings require a recertified calendar/field-packet promotion before becoming an executable menu change.
- Current-calendar skip-ready savings are only counted when the replacement owner route is already scheduled no later than the removed route.
- Partial shrink rows are deliberately unpriced until a regenerated replacement route card exists.
- This audit does not mark BTC progress or replace challenge-window activity validation.
