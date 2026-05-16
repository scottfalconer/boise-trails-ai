# Ownership Reassignment Optimization Audit

Generated: 2026-05-16T00:42:06Z
Status: `ownership_reassignment_reduces_existing_loop_work`

## Summary

- Routes audited: 47
- Ownership graph edges: 29
- Relevant components: 5
- Exact optimized relevant components: 5
- Reassigned official segments: 27 (11.38 mi)
- Order-free route cards: 25 -> 22 (3 removed)
- Order-free on-foot miles: 183.90 -> 179.32 (4.58 mi saved)
- Order-free p75/p90 saved: 280 / 315 min
- Current-calendar skip-ready savings: 0.00 mi, 0 p75 min, 0 p90 min
- Partial shrink routes needing regenerated cards: 10

## Components With Order-Free Savings

| Component | Routes | Reassigned ids | Removed routes | Saved on-foot mi | Saved p75 | Calendar note |
|---|---:|---:|---|---:|---:|---|
| C02 | 5 | 7 | 127-3: FD27C, 127-1: FD27A | 3.50 | 222 | 2 need reorder |
| C05 | 3 | 3 | 114-1: FD14A | 1.08 | 58 | 1 need reorder |

## Partial Shrink Credit Moves

| Route | Lost credit ids | Retained credit ids | Replacement owner(s) | Current on-foot mi |
|---|---|---|---|---:|
| 105-2: 4A | 1498 | 1499, 1500, 1690 | 123-1: 12 | 4.07 |
| 111-1: 14 | 1695 | 1560, 1561, 1562, 1595, 1685, 1694 | 104-1: FD04A | 10.74 |
| 115-1: 3 | 1593, 1594, 1629, 1630, 1631 | 1510, 1511, 1512, 1513, 1514, 1515, 1522, 1529, 1530, 1531, 1548, 1549, 1550, 1551, 1552, 1574, 1575, 1590, 1591, 1592, 1627, 1628, 1720 | 104-1: FD04A | 12.13 |
| 120-1: FD20A | 1564, 1683, 1684, 1685 | 1563, 1681, 1682 | 104-1: FD04A, 106-1: FD06A, 111-1: 14 | 13.10 |
| 122-1: FD22B | 1532, 1533 | 1534, 1535 | 119-2: FD19B, 123-1: 12 | 4.46 |
| 124-1: FD24A | 1706 | 1704, 1705, 1707, 1708 | 127-2: FD27B | 11.81 |
| 130-1: FD30A | 1687, 1688, 1689, 1708 | 1626, 1657, 1696 | 124-1: FD24A, 127-2: FD27B | 13.62 |
| 131-1: 18 | 1655, 1679 | 1501, 1502, 1503, 1678, 1703, 1721, 1732, 1733, 1734, 1735, 1736 | 125-1: FD25A, 125-2: FD25B | 11.25 |
| 116-2: 15B | 1619, 1624 | 1584, 1618, 1620, 1621, 1622, 1623 | 109-2: 10B | 4.87 |
| 118-1: FD18A | 1599, 1604 | 1597, 1598, 1600, 1601, 1602, 1603 | 114-2: FD14B | 13.32 |

## Scope Boundary

- This audit separates physical traversal from credit ownership; a route can physically stay unchanged while some official segment credit moves to a different route card.
- Order-free savings require a recertified calendar/field-packet promotion before becoming an executable menu change.
- Current-calendar skip-ready savings are only counted when the replacement owner route is already scheduled no later than the removed route.
- Partial shrink rows are deliberately unpriced until a regenerated replacement route card exists.
- This audit does not mark BTC progress or replace challenge-window activity validation.
