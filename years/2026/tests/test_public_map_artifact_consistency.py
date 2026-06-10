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


STALE_MENU_ROWS = [
    "| FD03A | 3h | Dry Creek Parking Area/Trailhead | 4.83 | 6.43 |",
    "| FD09A | 2h 13m | Dry Creek Parking Area/Trailhead | 1.44 | 3.96 |",
    "| FD14D | 1h 13m | Full Sail | 0.74 | 2.0 |",
]


CURRENT_MENU_TEXT = [
    "Upper Hulls Gulch: Scott's (4B)",
    "Hillside to Hollow: Full Sail (1A-2)",
    "Dry Creek: Sweet Connie (16A-1)",
]


CURRENT_PRIVATE_CANDIDATES_REDACTED = {
    "manual-16a-1",
    "manual-16a-2",
    "multi-start-16c-16c-ms-02-2-shingle-creek-trail",
}


def load_map_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_map_data_from_html(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    match = re.search(r"const DATA = (.*?);\nconst map =", html, flags=re.DOTALL)
    assert match, f"could not find embedded map data in {path}"
    return json.loads(match.group(1))


def component_key(component: dict) -> str:
    return str(component.get("field_menu_label") or component.get("candidate_id") or "")


def components_by_key(data: dict) -> dict:
    components = {}
    for package in data.get("packages") or []:
        for component in package.get("components") or []:
            key = component_key(component)
            if key:
                components[key] = component
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
        "segment_ids",
    ]
    return {key: component.get(key) for key in keys}


def collection_candidate_ids(data: dict, collection_name: str) -> set[str]:
    collection = (data.get("feature_collections") or {}).get(collection_name) or {}
    return {
        str((feature.get("properties") or {}).get("candidate_id"))
        for feature in collection.get("features") or []
    }


def test_public_map_data_matches_private_canonical_metrics_for_current_routes():
    private_components = components_by_key(load_map_data(PRIVATE_MAP_DATA))

    for public_path in [ROOT_MAP_DATA, EXAMPLE_MAP_DATA]:
        public_components = components_by_key(load_map_data(public_path))
        assert set(public_components) == set(private_components)
        for label, private in private_components.items():
            public = public_components[label]
            for key in [
                "candidate_id",
                "accepted_replacement_id",
                "route_card_status",
                "official_miles",
                "on_foot_miles",
                "total_minutes",
                "segment_ids",
            ]:
                assert public.get(key) == private.get(key), (public_path, label, key)


def test_public_html_embedded_data_matches_public_json_for_current_routes():
    root_json_components = components_by_key(load_map_data(ROOT_MAP_DATA))
    example_json_components = components_by_key(load_map_data(EXAMPLE_MAP_DATA))
    public_pairs = [
        (ROOT_MAP_HTML, root_json_components),
        (EXAMPLE_MAP_HTML, example_json_components),
    ]

    for html_path, json_components in public_pairs:
        html_components = components_by_key(extract_map_data_from_html(html_path))
        assert set(html_components) == set(json_components)
        for label in json_components:
            assert route_projection(html_components[label]) == route_projection(json_components[label])


def test_public_menu_markdown_contains_current_rows_and_not_stale_rows():
    for menu_path in [ROOT_MENU_MD, EXAMPLE_MENU_MD]:
        menu = menu_path.read_text(encoding="utf-8")
        assert "Open runnable outings: 31" in menu
        for expected_text in CURRENT_MENU_TEXT:
            assert expected_text in menu, (menu_path, expected_text)
        for stale_row in STALE_MENU_ROWS:
            assert stale_row not in menu, (menu_path, stale_row)


def test_public_map_data_redacts_private_anchors_but_keeps_current_public_features():
    data = load_map_data(ROOT_MAP_DATA)
    text = json.dumps(data)

    assert "Chukar Butte private Strava parking anchor" not in text
    assert "private Strava" not in text
    assert "Strava parking anchor" not in text
    assert set(data["public_sanitization"]["private_candidate_ids_redacted"]) == (
        CURRENT_PRIVATE_CANDIDATES_REDACTED
    )
    public_route_ids = collection_candidate_ids(data, "routes")
    public_parking_ids = collection_candidate_ids(data, "parking")
    assert CURRENT_PRIVATE_CANDIDATES_REDACTED.isdisjoint(public_route_ids)
    assert CURRENT_PRIVATE_CANDIDATES_REDACTED.isdisjoint(public_parking_ids)
    assert "multi-start-1a-1a-ms-04-2-full-sail-trail-buena-vista-trail-bob-smylie" in public_route_ids
    assert "multi-start-1a-1a-ms-04-2-full-sail-trail-buena-vista-trail-bob-smylie" in public_parking_ids
