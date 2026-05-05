import importlib.util
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
