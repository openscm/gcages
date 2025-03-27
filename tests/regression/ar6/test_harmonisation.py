"""
Test harmonisation compared to AR6

Note that you could use this to test all scenarios,
but we don't to save computational resources.
Instead, we just test a selection of scenarios
that cover all the code used in AR6.
"""

from __future__ import annotations

from pathlib import Path

from gcages.testing import KEY_TESTING_SCENARIOS, get_test_model_scenario_parameters

PROCESSED_AR6_DB_DIR = Path(__file__).parents[0] / "ar6-output-processed"


@get_test_model_scenario_parameters(
    model_scenarios=KEY_TESTING_SCENARIOS, processed_ar6_db_dir=PROCESSED_AR6_DB_DIR
)
def test_individual_scenario(model, scenario):
    assert False


def test_key_testing_scenarios_all_at_once():
    assert False
    KEY_TESTING_SCENARIOS
