import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PRIVATE_MAP_DATA = REPO_ROOT / "years/2026/outputs/private/2026-outing-menu-map-data.json"
ROOT_MAP_DATA = REPO_ROOT / "outing-menu-map-data.json"
EXAMPLE_MAP_DATA = REPO_ROOT / "years/2026/outputs/examples/2026-outing-menu-map-data.example.json"
ROOT_MAP_HTML = REPO_ROOT / "outing-menu-map.html"
EXAMPLE_MAP_HTML = REPO_ROOT / "years/2026/outputs/examples/2026-outing-menu-map.example.html"
ROOT_MENU_MD = REPO_ROOT / "outing-menu.md"
EXAMPLE_MENU_MD = REPO_ROOT / "years/2026/outputs/examples/2026-outing-menu.example.md"


REANCHORED_ROUTE_EXPECTATIONS = {
    "FD03A": {
        "candidate_id": "accepted-replacement-fd03a-chukar-butte-strava-anchor-19",
        "accepted_replacement_id": "fd03a-chukar-butte-strava-anchor-19",
        "route_card_status": "certified_route_card",
        "official_miles": 4.83,
        "on_foot_miles": 5.34,
        "total_minutes": 155,
        "public_trailhead": "Chukar Butte prior parking anchor",
    },
    "FD09A": {
        "candidate_id": "accepted-replacement-fd09a-barn-owl-west-hidden-springs",
        "accepted_replacement_id": "fd09a-barn-owl-west-hidden-springs-investigation",
        "route_card_status": "certified_route_card",
        "official_miles": 1.44,
        "on_foot_miles": 2.52,
        "total_minutes": 100,
        "public_trailhead": "West Hidden Springs Drive road-parking anchor",
    },
    "FD14D": {
        "candidate_id": "accepted-replacement-fd14d-36th-street-chute-lower-36th",
        "accepted_replacement_id": "fd14d-36th-street-chute-lower-36th",
        "route_card_status": "certified_route_card",
        "official_miles": 0.74,
        "on_foot_miles": 1.5,
        "total_minutes": 60,
        "public_trailhead": "Full Sail Trailhead, N 36th St Parking",
    },
}


EXPECTED_MENU_ROWS = {
    "FD03A": "| Dry Creek: Chukar Butte (FD03A) | 2h 35m | Chukar Butte prior parking anchor | 4.83 | 5.34 |",
    "FD09A": "| Dry Creek: Barn Owl (FD09A) | 1h 40m | West Hidden Springs Drive road-parking anchor | 1.44 | 2.52 |",
    "FD14D": "| Full Sail / N 36th St: 36th Street Chute (FD14D) | 1h | Full Sail Trailhead, N 36th St Parking | 0.74 | 1.5 |",
}


STALE_MENU_ROWS = [
    "| FD03A | 3h | Dry Creek Parking Area/Trailhead | 4.83 | 6.43 |",
    "| FD09A | 2h 13m | Dry Creek Parking Area/Trailhead | 1.44 | 3.96 |",
    "| FD14D | 1h 13m | Full Sail | 0.74 | 2.0 |",
]


def load_map_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_map_data_from_html(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    match = re.search(r"const DATA = (.*?);\nconst map =", html, flags=re.DOTALL)
    assert match, f"could not find embedded map data in {path}"
    return json.loads(match.group(1))


def components_by_label(data: dict) -> dict:
    components = {}
    for package in data.get("packages") or []:
        for component in package.get("components") or []:
            label = component.get("field_menu_label")
            if label:
                components[label] = component
    return components


def route_projection(component: dict) -> dict:
    keys = [
        "candidate_id",
        "accepted_replacement_id",
        "route_card_status",
        "official_miles",
        "on_foot_miles",
        "total_minutes",
        "trailhead",
    ]
    return {key: component.get(key) for key in keys}


def collection_candidate_ids(data: dict, collection_name: str) -> set[str]:
    collection = (data.get("feature_collections") or {}).get(collection_name) or {}
    return {
        str((feature.get("properties") or {}).get("candidate_id"))
        for feature in collection.get("features") or []
    }


def test_public_map_data_matches_private_canonical_metrics_for_reanchored_routes():
    private_components = components_by_label(load_map_data(PRIVATE_MAP_DATA))

    for public_path in [ROOT_MAP_DATA, EXAMPLE_MAP_DATA]:
        public_components = components_by_label(load_map_data(public_path))
        for label, expected in REANCHORED_ROUTE_EXPECTATIONS.items():
            private = private_components[label]
            public = public_components[label]
            for key in [
                "candidate_id",
                "accepted_replacement_id",
                "route_card_status",
                "official_miles",
                "on_foot_miles",
                "total_minutes",
            ]:
                assert public[key] == private[key] == expected[key], (public_path, label, key)
            assert public["trailhead"] == expected["public_trailhead"], (public_path, label)


def test_public_html_embedded_data_matches_public_json_for_reanchored_routes():
    root_json_components = components_by_label(load_map_data(ROOT_MAP_DATA))
    example_json_components = components_by_label(load_map_data(EXAMPLE_MAP_DATA))
    public_pairs = [
        (ROOT_MAP_HTML, root_json_components),
        (EXAMPLE_MAP_HTML, example_json_components),
    ]

    for html_path, json_components in public_pairs:
        html_components = components_by_label(extract_map_data_from_html(html_path))
        for label in REANCHORED_ROUTE_EXPECTATIONS:
            assert route_projection(html_components[label]) == route_projection(json_components[label])


def test_public_menu_markdown_contains_reanchored_rows_and_not_stale_rows():
    for menu_path in [ROOT_MENU_MD, EXAMPLE_MENU_MD]:
        menu = menu_path.read_text(encoding="utf-8")
        for expected_row in EXPECTED_MENU_ROWS.values():
            assert expected_row in menu, (menu_path, expected_row)
        for stale_row in STALE_MENU_ROWS:
            assert stale_row not in menu, (menu_path, stale_row)


def test_public_map_data_redacts_private_anchor_but_keeps_public_safe_fd14d_features():
    data = load_map_data(ROOT_MAP_DATA)
    text = json.dumps(data)
    fd03a = REANCHORED_ROUTE_EXPECTATIONS["FD03A"]["candidate_id"]
    fd09a = REANCHORED_ROUTE_EXPECTATIONS["FD09A"]["candidate_id"]
    fd14d = REANCHORED_ROUTE_EXPECTATIONS["FD14D"]["candidate_id"]

    assert "Chukar Butte private Strava parking anchor" not in text
    assert "private Strava" not in text
    assert "Strava parking anchor" not in text
    assert fd03a in data["public_sanitization"]["private_candidate_ids_redacted"]
    assert fd14d not in data["public_sanitization"]["private_candidate_ids_redacted"]
    assert fd14d in collection_candidate_ids(data, "routes")
    assert fd14d in collection_candidate_ids(data, "parking")
    assert fd09a in collection_candidate_ids(data, "routes")
    assert fd09a in collection_candidate_ids(data, "parking")
    assert fd03a not in collection_candidate_ids(data, "routes")
    assert fd03a not in collection_candidate_ids(data, "parking")
