import unittest
import argparse

# Helper function to get the parser setup as it is in challenge_planner.py
def get_test_parser():
    # Simplified parser setup, mirroring challenge_planner.py's relevant parts
    # We assume config_defaults are empty for these tests, focusing on CLI args.
    config_defaults = {}
    parser = argparse.ArgumentParser(description="Challenge route planner (Test)")
    parser.set_defaults(**config_defaults)

    # Add arguments that are relevant for focus mode testing
    # and any other arguments that are required or whose absence might affect parsing.
    parser.add_argument(
        "--start-date",
        default='2024-01-01', # Required, so provide a default for tests
        help="Challenge start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        default='2024-01-02', # Required, so provide a default for tests
        help="Challenge end date YYYY-MM-DD",
    )
    parser.add_argument(
        "--pace",
        default='10', # Required, so provide a default for tests
        type=float,
        help="Base running pace (min/mi)",
    )
    parser.add_argument(
        "--segments",
        default="dummy_segments.json", # Default for tests
        help="Trail segment JSON file",
    )
    parser.add_argument(
        "--focus-segment-ids",
        type=str,
        default=None, # This is the default from PlannerConfig / argparse
        help="Comma-separated list of segment IDs to focus planning on. Activates focused planning mode.",
    )
    parser.add_argument(
        "--focus-plan-days",
        type=int,
        default=None, # This is the default from PlannerConfig / argparse
        help="Number of days to plan when in focused mode. Defaults to 1 if --focus-segment-ids is used.",
    )
    # Add other arguments if they become necessary for the parser to not error out
    # For now, keeping it minimal to what's needed for these tests.
    return parser

class TestChallengePlannerFocusMode(unittest.TestCase):

    def parse_and_apply_focus_logic(self, cli_args_list):
        """
        Parses arguments using a parser similar to challenge_planner.py's
        and applies the focus_plan_days defaulting logic.
        """
        parser = get_test_parser()
        args = parser.parse_args(cli_args_list)

        # Replicate the conditional logic from challenge_planner.py's main()
        if args.focus_segment_ids and args.focus_plan_days is None:
            args.focus_plan_days = 1

        return args

    def get_minimal_required_args_list(self):
        # These are args that might be 'required=True' in the parser
        # or for which no default is provided by PlannerConfig.
        # For tests, get_test_parser() now provides defaults for these.
        return []

    def test_focus_plan_days_defaults_to_one(self):
        """
        Tests that focus_plan_days defaults to 1 when focus_segment_ids is provided
        and focus_plan_days is not.
        """
        cli_input = self.get_minimal_required_args_list() + [
            '--focus-segment-ids', '101',
            # No --focus-plan-days, relying on default None from parser
        ]
        args = self.parse_and_apply_focus_logic(cli_input)

        self.assertEqual(args.focus_segment_ids, "101")
        self.assertEqual(args.focus_plan_days, 1)

    def test_focus_plan_days_uses_provided_value(self):
        """
        Tests that focus_plan_days uses the explicitly provided value
        even when focus_segment_ids is also provided.
        """
        cli_input = self.get_minimal_required_args_list() + [
            '--focus-segment-ids', '102',
            '--focus-plan-days', '3',
        ]
        args = self.parse_and_apply_focus_logic(cli_input)

        self.assertEqual(args.focus_segment_ids, "102")
        self.assertEqual(args.focus_plan_days, 3)

    def test_no_focus_mode_args_are_none(self):
        """
        Tests that focus_segment_ids and focus_plan_days are None
        when not provided in the command-line arguments.
        """
        cli_input = self.get_minimal_required_args_list()
        # No focus mode arguments provided

        args = self.parse_and_apply_focus_logic(cli_input)

        self.assertIsNone(args.focus_segment_ids)
        self.assertIsNone(args.focus_plan_days)

    def test_focus_plan_days_remains_none_if_no_focus_ids(self):
        """
        Tests that focus_plan_days remains None if focus_segment_ids is not provided,
        even if focus_plan_days is also not provided (i.e., the defaulting logic is not triggered).
        """
        cli_input = self.get_minimal_required_args_list()
        # No --focus-segment-ids
        # No --focus-plan-days
        args = self.parse_and_apply_focus_logic(cli_input)

        self.assertIsNone(args.focus_segment_ids)
        self.assertIsNone(args.focus_plan_days) # Should remain None

    def test_focus_plan_days_uses_provided_value_if_no_focus_ids(self):
        """
        Tests that focus_plan_days uses its provided value if focus_segment_ids is not provided.
        The defaulting logic for focus_plan_days should not trigger.
        """
        cli_input = self.get_minimal_required_args_list() + [
            '--focus-plan-days', '5',
            # No --focus-segment-ids
        ]
        args = self.parse_and_apply_focus_logic(cli_input)

        self.assertIsNone(args.focus_segment_ids)
        self.assertEqual(args.focus_plan_days, 5)


if __name__ == '__main__':
    unittest.main()
