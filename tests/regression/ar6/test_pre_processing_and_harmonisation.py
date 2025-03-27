"""
Test pre-processing and harmonisation compared to AR6

Note that you could use this to test all scenarios,
but we don't to save computational resources.
Instead, we just test a selection of scenarios
that cover all the code used in AR6.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gcages.testing import (
    KEY_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_ar6_harmonised_emissions,
    get_ar6_raw_emissions,
    get_key_testing_model_scenario_parameters,
)

pix = pytest.importorskip("pandas_indexing")

PROCESSED_AR6_DB_DIR = Path(__file__).parents[0] / "ar6-output-processed"


@get_key_testing_model_scenario_parameters()
def test_individual_scenario(model, scenario):
    raw = get_ar6_raw_emissions(
        model=model,
        scenario=scenario,
        processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
    )
    if raw.empty:
        msg = f"No test data for {model=} {scenario=}?"
        raise AssertionError(msg)

    pre_processor = AR6PreProcessor.from_ar6_like_config(
        run_checks=False, n_processes=1
    )
    harmoniser = AR6Harmoniser.from_ar6_like_config(run_checks=False, n_processes=1)

    pre_processed = pre_processor(raw)
    res = harmoniser(pre_processed)

    exp = (
        get_ar6_harmonised_emissions(
            model=model,
            scenario=scenario,
            processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
        )
        # Ignore aggregate stuff
        .loc[~pix.ismatch(variable=["**Kyoto**", "**F-Gases", "**HFC", "**PFC"])]
    )

    assert_frame_equal(res, exp)


def test_key_testing_scenarios_all_at_once():
    assert False
    KEY_TESTING_MODEL_SCENARIOS
