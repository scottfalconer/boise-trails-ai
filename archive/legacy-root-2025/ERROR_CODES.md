# Planner Error Codes

This document explains the exit codes and common error messages produced by the command line tools in **boise-trails-ai**. Use this as a reference when troubleshooting unexpected planner failures.

## Exit Codes

| Code | Meaning |
|-----:|---------|
| `0`  | Planner finished successfully and all routes were generated. |
| `1`  | Planning failed. One or more errors prevented a complete plan from being produced. |

The planner returns `1` whenever it cannot schedule every required segment (for example if time budgets are too small) or when a critical exception is encountered while loading data files or computing routes.

## Common Error Messages

| Message | Description |
|---------|-------------|
| `DijkstraTimeoutError` | Dijkstra pathfinding exceeded the configured time limit. Consider increasing the timeout or simplifying the route. |
| `Failed to open RocksDB` | The RocksDB cache could not be opened. The planner falls back to an in-memory cache but performance may suffer. |
| `Configuration file must contain a mapping` | The supplied config file does not parse into key/value pairs. Check that it is valid JSON or YAML. |
| `Unrecognized segment JSON structure` | The segment data file does not match the expected schema. Verify you are using the provided challenge dataset. |
| `No track points` | A GPX file contained no track data when running `gpx_to_csv`. |
| `Routing errors detected. Resolve them before exporting the plan.` | The planner could not build a feasible schedule for all segments. Increase daily time or adjust parameters. |
| `Error: The following ... challenge segments were not scheduled` | Specific segment IDs were not included in the generated plan. Review the debug log to see why they were skipped. |

Additional error messages may appear when reading files or performing network operations. When troubleshooting, running the planner with the `--verbose` flag will echo these messages to the console and help pinpoint failures.
