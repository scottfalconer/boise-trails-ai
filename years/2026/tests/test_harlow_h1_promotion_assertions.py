import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "harlow_h1_promotion_assertions.py"


def load_module():
    spec = importlib.util.spec_from_file_location("harlow_h1_promotion_assertions", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_route_count_uses_promotion_summary_before_falling_back_to_old_count():
    module = load_module()

    assert module.expected_route_count_from_promotion(
        {"summary": {"expected_active_route_cards_after_export": 43, "old_route_card_count": 47}}
    ) == 43
    assert module.expected_route_count_from_promotion({"summary": {"old_route_card_count": 47}}) == 43
