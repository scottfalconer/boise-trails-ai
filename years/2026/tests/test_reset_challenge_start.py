import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "reset_challenge_start.py"


def load_reset_module():
    spec = importlib.util.spec_from_file_location("reset_challenge_start", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reset_state_fields_clears_progress_and_preserves_personal_settings():
    module = load_reset_module()
    state = {
        "completed_segment_ids": [1, 2],
        "blocked_segment_ids": [3],
        "blocked_trail_names": ["Closed Trail"],
        "pace_min_per_mile": 15.46,
        "drive_model": {"origin_label": "Home"},
    }

    reset = module.reset_state_fields(state)

    assert reset["completed_segment_ids"] == []
    assert reset["blocked_segment_ids"] == []
    assert reset["blocked_trail_names"] == []
    assert reset["pace_min_per_mile"] == 15.46
    assert reset["drive_model"] == {"origin_label": "Home"}


def test_reset_state_fields_can_preserve_current_blocks():
    module = load_reset_module()
    state = {
        "completed_segment_ids": [1, 2],
        "blocked_segment_ids": [3],
        "blocked_trail_names": ["Closed Trail"],
    }

    reset = module.reset_state_fields(state, preserve_blocks=True)

    assert reset["completed_segment_ids"] == []
    assert reset["blocked_segment_ids"] == [3]
    assert reset["blocked_trail_names"] == ["Closed Trail"]


def test_build_pipeline_commands_regenerates_canonical_private_map_chain(tmp_path):
    module = load_reset_module()
    state_json = tmp_path / "state.private.json"
    replacements_json = tmp_path / "replacements.json"

    commands = module.build_pipeline_commands(state_json, field_menu_replacements_json=replacements_json)
    command_text = [" ".join(command) for command in commands]

    assert command_text[0].startswith(sys.executable)
    assert "personal_route_planner.py" in command_text[0]
    assert f"--state {state_json}" in command_text[0]
    assert "years/2026/outputs/private/personal-route-menu.json" in command_text[0]
    assert any("block_route_candidate_pass.py" in command for command in command_text)
    assert any("block_combo_route_pass.py" in command for command in command_text)
    assert any("block_route_assembler.py" in command for command in command_text)
    assert any("block_hybrid_route_pass.py" in command for command in command_text)
    assert any("manual_route_design_pass.py" in command for command in command_text)
    assert any("multi_start_alternative_audit.py" in command for command in command_text)
    assert any("multi_start_field_menu_replacements.py" in command for command in command_text)
    assert command_text[-4].endswith(
        f"human_loop_plan.py --field-menu-overrides-json {module.DEFAULT_FIELD_MENU_OVERRIDES_JSON}"
    )
    assert "years/2026/scripts/human_loop_plan.py" in command_text[-1]
    assert f"--field-menu-overrides-json {replacements_json}" in command_text[-1]
    assert module.DEFAULT_MAP_DATA_JSON.name == "2026-outing-menu-map-data.json"


def test_build_pipeline_commands_uses_generated_multi_start_replacements_by_default(tmp_path, monkeypatch):
    module = load_reset_module()
    replacements = tmp_path / "generated-replacements.private.json"
    monkeypatch.setattr(module, "DEFAULT_GENERATED_MULTI_START_FIELD_MENU_REPLACEMENTS_JSON", replacements)

    commands = module.build_pipeline_commands(tmp_path / "state.private.json")
    assert str(replacements) in " ".join(commands[-1])


def test_reset_record_can_include_locked_original_snapshot(tmp_path):
    module = load_reset_module()

    record = module.build_reset_record(
        reset_at="20260509T000000Z",
        state_json=tmp_path / "state.private.json",
        backup_path=None,
        preserve_blocks=False,
        pipeline_results=[],
        output_verification={"clean_start": True},
        locked_original={"epoch": "challenge-2026", "state_snapshot_json": "/tmp/original/state.json"},
    )

    assert record["locked_original"]["epoch"] == "challenge-2026"
    assert record["locked_original"]["state_snapshot_json"] == "/tmp/original/state.json"
