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

    commands = module.build_pipeline_commands(state_json)
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
    assert command_text[-1].endswith("years/2026/scripts/human_loop_plan.py")
    assert module.DEFAULT_MAP_DATA_JSON.name == "2026-outing-menu-map-data.json"
