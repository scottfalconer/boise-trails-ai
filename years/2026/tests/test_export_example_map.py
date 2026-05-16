import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_example_map.py"


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_example_map", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sanitize_map_html_removes_local_private_paths(tmp_path):
    module = load_exporter()
    repo_root = tmp_path / "boise-trails-ai"
    html = (
        '<script>const DATA = {"map_html_path":"'
        + str(repo_root)
        + '/years/2026/outputs/private/route-blocks/example.html",'
        + '"gpx_path":"'
        + str(repo_root)
        + '/years/2026/outputs/private/route-blocks/example.gpx"};</script>'
    )

    sanitized = module.sanitize_map_html(html, repo_root=repo_root)

    assert str(repo_root) not in sanitized
    assert "outputs/private" not in sanitized
    assert "years/2026/outputs/example-redacted/route-blocks/example.html" in sanitized
    assert "years/2026/outputs/example-redacted/route-blocks/example.gpx" in sanitized


def test_sanitize_menu_markdown_rewrites_private_map_path(tmp_path):
    module = load_exporter()
    repo_root = tmp_path / "boise-trails-ai"
    markdown = (
        "# 2026 Outing Menu\n\n"
        "- Map: `"
        + str(repo_root)
        + "/years/2026/outputs/private/2026-outing-menu-map.html`\n"
        "- Detail: `"
        + str(repo_root)
        + "/years/2026/outputs/private/route-blocks/example.gpx`\n"
    )

    sanitized = module.sanitize_menu_markdown(
        markdown, map_link="outing-menu-map.html", repo_root=repo_root
    )

    assert str(repo_root) not in sanitized
    assert "outputs/private" not in sanitized
    assert "- Map: `outing-menu-map.html`" in sanitized
    assert "years/2026/outputs/example-redacted/route-blocks/example.gpx" in sanitized


def test_sanitize_menu_markdown_rewrites_private_anchor_labels():
    module = load_exporter()
    markdown = (
        "| 1A-1 | 54 min | Strava parking anchor 13 |\n"
        "| 4C-2 | 3h 8m | Strava parking anchor 21 |\n"
        "| FD03A | 2h 35m | Chukar Butte private Strava parking anchor |\n"
    )

    sanitized = module.sanitize_menu_markdown(markdown, map_link="outing-menu-map.html")

    assert "Strava parking anchor" not in sanitized
    assert "private Strava" not in sanitized
    assert "Full Sail Trailhead, N 36th St Parking" in sanitized
    assert "Castle Rock-side prior parking anchor" in sanitized
    assert "Chukar Butte prior parking anchor" in sanitized


def test_sanitize_private_map_data_preserves_public_safe_fd14d_and_redacts_private_anchor():
    module = load_exporter()
    fd14d = "accepted-replacement-fd14d-36th-street-chute-lower-36th"
    fd03a = "accepted-replacement-fd03a-chukar-butte-strava-anchor-19"
    data = {
        "packages": [
            {
                "primary_trailhead": "Chukar Butte private Strava parking anchor",
                "trailheads": [
                    "Full Sail Trailhead, N 36th St Parking",
                    "Chukar Butte private Strava parking anchor",
                ],
                "components": [
                    {
                        "field_menu_label": "FD14D",
                        "candidate_id": fd14d,
                        "trailhead": "Full Sail Trailhead, N 36th St Parking",
                        "accepted_anchor_label": "Full Sail Trailhead, N 36th St Parking",
                    },
                    {
                        "field_menu_label": "FD03A",
                        "candidate_id": fd03a,
                        "trailhead": "Chukar Butte private Strava parking anchor",
                        "accepted_anchor_label": "Chukar Butte private Strava parking anchor",
                        "start_justification": (
                            "Chosen because the private Strava-derived parking anchor "
                            "is accepted for exact Chukar Butte credit."
                        ),
                    },
                ],
            }
        ],
        "route_cues": {
            fd14d: {
                "trailhead": {
                    "name": "Full Sail Trailhead, N 36th St Parking",
                    "lat": 43.6617,
                    "lon": -116.2266,
                    "source": "strava_activity_endpoint_cluster",
                    "parking_confidence": "strava_seen_prior_challenge_window",
                }
            },
            fd03a: {
                "trailhead": {
                    "name": "Chukar Butte private Strava parking anchor",
                    "lat": 43.7213,
                    "lon": -116.2344,
                    "source": "strava_activity_endpoint_cluster",
                    "parking_confidence": "strava_seen_prior_challenge_window",
                },
                "start_justification": (
                    "Chosen because the private Strava-derived parking anchor is accepted."
                ),
            },
        },
        "feature_collections": {
            "routes": {
                "features": [
                    {"properties": {"candidate_id": fd14d, "trailhead": "Full Sail Trailhead, N 36th St Parking"}},
                    {"properties": {"candidate_id": fd03a, "trailhead": "Chukar Butte private Strava parking anchor"}},
                ]
            },
            "parking": {
                "features": [
                    {
                        "properties": {
                            "candidate_id": fd14d,
                            "name": "Full Sail Trailhead, N 36th St Parking",
                            "source": "strava_activity_endpoint_cluster",
                        }
                    },
                    {
                        "properties": {
                            "candidate_id": fd03a,
                            "name": "Chukar Butte private Strava parking anchor",
                            "source": "strava_activity_endpoint_cluster",
                        }
                    },
                ]
            },
            "logistics": {
                "features": [
                    {"properties": {"candidate_id": fd14d}},
                    {"properties": {"candidate_id": fd03a}},
                ]
            },
            "official_segments": {
                "features": [
                    {
                        "properties": {
                            "candidate_id": fd03a,
                            "trailhead": "Chukar Butte private Strava parking anchor",
                        }
                    }
                ]
            },
        },
    }

    sanitized = module.sanitize_private_map_data(data)

    assert fd14d not in sanitized["public_sanitization"]["private_candidate_ids_redacted"]
    assert fd03a in sanitized["public_sanitization"]["private_candidate_ids_redacted"]
    assert sanitized["route_cues"][fd14d]["trailhead"]["lat"] == 43.6617
    assert sanitized["route_cues"][fd14d]["trailhead"]["source"] == "accepted_prior_activity_anchor"
    assert sanitized["route_cues"][fd03a]["trailhead"] == {
        "name": "Chukar Butte prior parking anchor",
        "privacy": "private_coordinates_redacted",
    }
    route_ids = {
        feature["properties"]["candidate_id"]
        for feature in sanitized["feature_collections"]["routes"]["features"]
    }
    assert route_ids == {fd14d}
    assert sanitized["packages"][0]["primary_trailhead"] == "Chukar Butte prior parking anchor"
    assert sanitized["packages"][0]["components"][1]["accepted_anchor_label"] == (
        "Chukar Butte prior parking anchor"
    )
    assert "Strava" not in json.dumps(sanitized)


def test_sanitize_map_data_json_exports_same_payload_without_private_paths(tmp_path):
    module = load_exporter()
    repo_root = tmp_path / "boise-trails-ai"
    html = (
        '<script>\nconst DATA = {"map_html_path":"'
        + str(repo_root)
        + '/years/2026/outputs/private/2026-outing-menu-map.html","safe":"value"};'
        + '\nconst map = {};\n</script>'
    )

    sanitized = module.sanitize_map_data_json(html, repo_root=repo_root)
    data = json.loads(sanitized)

    assert data["safe"] == "value"
    assert str(repo_root) not in sanitized
    assert "outputs/private" not in sanitized
    assert data["map_html_path"] == "years/2026/outputs/example-redacted/2026-outing-menu-map.html"
