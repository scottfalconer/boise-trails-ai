import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "artifact_utils.py"


def load_artifacts():
    spec = importlib.util.spec_from_file_location("artifact_utils", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_record_includes_size_and_sha256(tmp_path):
    artifacts = load_artifacts()
    path = tmp_path / "example.json"
    path.write_text('{"ok": true}\n', encoding="utf-8")

    record = artifacts.artifact_record(path)

    assert record["path"] == str(path)
    assert record["exists"] is True
    assert record["size_bytes"] == len('{"ok": true}\n')
    assert len(record["sha256"]) == 64


def test_verify_manifest_fails_when_artifact_changes(tmp_path):
    artifacts = load_artifacts()
    path = tmp_path / "output.json"
    path.write_text('{"version": 1}\n', encoding="utf-8")
    manifest = artifacts.build_artifact_manifest(
        run_id="test-run",
        inputs=[],
        outputs=[path],
    )

    assert artifacts.verify_artifact_manifest(manifest)["valid"] is True

    path.write_text('{"version": 2}\n', encoding="utf-8")
    result = artifacts.verify_artifact_manifest(manifest)

    assert result["valid"] is False
    assert result["mismatches"][0]["path"] == str(path)
