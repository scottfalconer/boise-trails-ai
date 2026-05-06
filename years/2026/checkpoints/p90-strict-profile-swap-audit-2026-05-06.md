# Strict Profile Swap Audit

Objective: measure opportunity cost of forcing each strict-profile missing segment into the 31-day max-coverage schedule

## Summary

- Scenario: `strict_current_p90_bounds`
- Bounds: 260 weekday / 180 weekend
- Day counts: 22 weekday / 9 weekend
- Baseline max coverage: 219 covered / 32 missing
- Baseline official miles: 122.45 covered / 41.98 missing
- Forced missing segments tested: 32
- Classification counts: `{'coverage_loss_swap': 21, 'no_strict_field_day_candidate': 1, 'one_for_one_swap': 10}`

## Best Review Rows

| Segment | Trail | Classification | Delta covered | Delta p75 | Recovers | Loses |
|---:|---|---|---:|---:|---|---|
| 1655 | Shindig 2 | one_for_one_swap | 0 | -4 | 1655 | 1703 |
| 1707 | Harlow's Hollows 2 | one_for_one_swap | 0 | 2 | 1661, 1707 | 1626, 1703 |
| 1657 | Shooting Range 1 | one_for_one_swap | 0 | 3 | 1657, 1708 | 1626, 1703 |
| 1708 | Harlow's Hollows Connector 1 | one_for_one_swap | 0 | 3 | 1657, 1708 | 1626, 1703 |
| 1750 | Around the Mountain Trail 7 | one_for_one_swap | 0 | 4 | 1750 | 1703 |
| 1721 | Lodge Trail 1 | one_for_one_swap | 0 | 10 | 1721 | 1703 |
| 1689 | Twisted Spring 3 | one_for_one_swap | 0 | 32 | 1657, 1689 | 1558, 1652 |
| 1661 | Spring Creek 1 | one_for_one_swap | 0 | 34 | 1657, 1661 | 1558, 1652 |
| 1542 | Dry Creek Trail 1 | one_for_one_swap | 0 | 54 | 1542 | 1516 |
| 1543 | Dry Creek Trail 2 | one_for_one_swap | 0 | 59 | 1543, 1657, 1661 | 1488, 1516, 1703 |
| 1660 | Sidewinder Trail 1 | coverage_loss_swap | -1 | -20 | 1660 | 1558, 1652 |
| 1662 | Spring Creek 2 | coverage_loss_swap | -1 | -11 | 1662 | 1516, 1706 |

## All Forced-Segment Rows

| Segment | Trail | Official mi | Options | Classification | Delta covered | Delta p75 | Lost covered count |
|---:|---|---:|---:|---|---:|---:|---:|
| 1655 | Shindig 2 | 0.12 | 15 | one_for_one_swap | 0 | -4 | 1 |
| 1707 | Harlow's Hollows 2 | 0.39 | 21 | one_for_one_swap | 0 | 2 | 2 |
| 1657 | Shooting Range 1 | 0.28 | 0 | one_for_one_swap | 0 | 3 | 2 |
| 1708 | Harlow's Hollows Connector 1 | 0.88 | 24 | one_for_one_swap | 0 | 3 | 2 |
| 1750 | Around the Mountain Trail 7 | 0.34 | 12 | one_for_one_swap | 0 | 4 | 1 |
| 1721 | Lodge Trail 1 | 0.54 | 12 | one_for_one_swap | 0 | 10 | 1 |
| 1689 | Twisted Spring 3 | 0.07 | 164 | one_for_one_swap | 0 | 32 | 2 |
| 1661 | Spring Creek 1 | 0.08 | 158 | one_for_one_swap | 0 | 34 | 2 |
| 1542 | Dry Creek Trail 1 | 0.58 | 22 | one_for_one_swap | 0 | 54 | 1 |
| 1543 | Dry Creek Trail 2 | 0.74 | 19 | one_for_one_swap | 0 | 59 | 3 |
| 1660 | Sidewinder Trail 1 | 1.34 | 8 | coverage_loss_swap | -1 | -20 | 2 |
| 1662 | Spring Creek 2 | 2.33 | 8 | coverage_loss_swap | -1 | -11 | 2 |
| 1597 | Peggy's Trail 1 | 4.57 | 4 | coverage_loss_swap | -1 | -2 | 2 |
| 1709 | Cartwright Connector 1 | 1.7 | 14 | coverage_loss_swap | -1 | 13 | 2 |
| 1489 | Around the Mountain Trail 2 | 0.53 | 6 | coverage_loss_swap | -1 | 31 | 2 |
| 1665 | Sweet Connie Trail 1 | 0.84 | 2 | coverage_loss_swap | -1 | 32 | 2 |
| 1540 | Deer Point Trail 1 | 1.14 | 5 | coverage_loss_swap | -1 | 36 | 2 |
| 1705 | Harlow's Hollows 3 | 0.48 | 28 | coverage_loss_swap | -1 | 37 | 2 |
| 1666 | Sweet Connie Trail 2 | 0.71 | 2 | coverage_loss_swap | -1 | 40 | 2 |
| 1680 | The Face Trail 1 | 1.15 | 4 | coverage_loss_swap | -1 | 41 | 2 |
| 1492 | Around the Mountain Trail 5 | 0.44 | 1 | coverage_loss_swap | -2 | -11 | 3 |
| 1493 | Around the Mountain Trail 6 | 0.87 | 1 | coverage_loss_swap | -2 | -9 | 3 |
| 1731 | Cervidae Peak 1 | 2.24 | 3 | coverage_loss_swap | -2 | -3 | 3 |
| 1667 | Sweet Connie Trail 3 | 4.53 | 2 | coverage_loss_swap | -2 | -1 | 3 |
| 1545 | Dry Creek Trail 4 | 3.02 | 2 | coverage_loss_swap | -2 | 5 | 3 |
| 1653 | Sheep Camp Trail 1 | 0.77 | 1 | coverage_loss_swap | -2 | 9 | 3 |
| 1490 | Around the Mountain Trail 3 | 2.0 | 1 | coverage_loss_swap | -2 | 24 | 3 |
| 1491 | Around the Mountain Trail 4 | 1.7 | 1 | coverage_loss_swap | -2 | 41 | 3 |
| 1704 | Harlow's Hollows 4 | 0.2 | 1 | coverage_loss_swap | -2 | 42 | 3 |
| 1546 | Dry Creek Trail 5 | 1.65 | 1 | coverage_loss_swap | -2 | 61 | 3 |
| 1544 | Dry Creek Trail 3 | 0.99 | 1 | coverage_loss_swap | -2 | 65 | 3 |
| 1656 | Shingle Creek Trail 1 | 4.76 | 0 | no_strict_field_day_candidate |  |  | 0 |

## Interpretation

- one_for_one_swap means the segment can be forced without reducing total segment count, but another currently covered segment drops out.
- coverage_loss_swap means forcing the segment lowers total strict-profile max coverage and needs better grouping or a larger bound.
- no_strict_field_day_candidate means route/access/time redesign is needed before the segment can even compete in the strict schedule.
