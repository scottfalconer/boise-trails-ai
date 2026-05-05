# Boise Trails Challenge AI Planner

*An automated route planner to help plan an optimal route for the Boise Trails Challenge.*

The [Boise Trails Challenge](https://boisetrailschallenge.com) is a month-long event where participants attempt to cover **every official trail segment** in the Boise area. This project provides a sophisticated VRP-based planner to generate an optimal, multi-day hiking plan that covers all required segments with minimal on-foot mileage.

For a detailed technical breakdown of the formal problem definition (CARP, WPP, etc.) and the agent's development history, please see [`AGENTS.md`](AGENTS.md). For the system architecture and design decisions, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## How It Works

The daily planner uses Google's OR-Tools to solve the complex routing problem. Here's a high-level overview:

1.  **Build Master Graph**: The planner loads all trail segments (both required and optional) and constructs a single, connected graph of the entire trail network. It "heals" the graph to connect nearby but disconnected trailheads.
2.  **VRP Formulation & Solving**: The problem is then handed to the OR-Tools solver, configured with your available hiking days (e.g., short, medium, long days) and the trailheads you can park at, as defined in `config/daily_planner_config.yaml`.
3.  **Generate Plan**: The solver finds the most efficient set of routes to cover all required trails. The planner then decodes this solution, intelligently splitting the routes into practical, individual hikes. If the path between two trails is too long to walk, it assumes you will drive, creating a new hike starting from a closer trailhead.

## Using the Planner

### 1. Configure Your Plan
Edit `config/daily_planner_config.yaml` to set your available hiking days, their distance limits, and the list of trailheads you can park at.

### 2. Run the Planner
```bash
python -m src.trail_route_ai.daily_planner
```

### 3. Get Your Results
The planner will output:
*   `output/daily_plan_summary.csv`: A detailed list of all the hikes, showing which day to do them, where to park, the on-foot mileage, and which required segments are covered.
*   `output/routes/`: A directory containing a GPX file for each individual hike, ready to be loaded onto your GPS device.

## Installation

Before using the planner, set up the Python environment and dependencies:

1. **Clone the repository** (or download the source).
2. **Install dependencies**. It's recommended to use a Python virtual environment. For example:

   ```bash
   python -m venv .venv  
   source .venv/bin/activate  
   pip install -r requirements.txt
   ```

3. **Install the project in editable mode**:
   ```bash
   pip install -e .
   ```

## Download Data Assets

Run the helper script to fetch external data, such as the Digital Elevation Model (DEM) for elevation-aware routing:

```bash
bash run/get_data.sh
```
This will place required assets under the `data/` directory.

## Running Tests

To run the test suite for development purposes:
```bash
pip install -r requirements-dev.txt
pytest
```
The test suite includes a validation test that confirms the generated plan covers all required segments.