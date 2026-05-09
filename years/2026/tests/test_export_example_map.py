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
    markdown = "| 1A-1 | 54 min | Strava parking anchor 13 |\n| 4C-2 | 3h 8m | Strava parking anchor 21 |\n"

    sanitized = module.sanitize_menu_markdown(markdown, map_link="outing-menu-map.html")

    assert "Strava parking anchor" not in sanitized
    assert "Full Sail Trailhead, N 36th St Parking" in sanitized
    assert "Castle Rock-side prior parking anchor" in sanitized


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
