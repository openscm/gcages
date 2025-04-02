"""
Test infilling compared to AR6

Note that you could use this to test all scenarios,
but we don't to save computational resources.
Instead, we just test a selection of scenarios
that cover the key paths from AR6.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gcages.ar6 import AR6Infiller
from gcages.testing import (
    KEY_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_ar6_harmonised_emissions,
    get_ar6_infilled_emissions,
    get_key_testing_model_scenario_parameters,
)

pix = pytest.importorskip("pandas_indexing")
# Only works if silicone installed
pytest.importorskip("silicone")

AR6_INFILLING_DB_FILE = (
    Path(__file__).parents[0] / "ar6-workflow-inputs" / "infilling_db_ar6.csv"
)
AR6_INFILLING_DB_CFCS_FILE = (
    Path(__file__).parents[0] / "ar6-workflow-inputs" / "infilling_db_ar6_cfcs.csv"
)
PROCESSED_AR6_DB_DIR = Path(__file__).parents[0] / "ar6-output-processed"


@get_key_testing_model_scenario_parameters()
@pytest.mark.slow
def test_individual_scenario(model, scenario):
    harmonised = (
        get_ar6_harmonised_emissions(
            model=model,
            scenario=scenario,
            processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
        )
        # Ignore aggregate stuff
        # (but keep CO2 total, that is needed)
        .loc[~pix.ismatch(variable=["**Kyoto**", "**F-Gases", "**HFC", "**PFC"])]
    )
    if harmonised.empty:
        msg = f"No harmonised data for {model=} {scenario=}?"
        raise AssertionError(msg)

    infiller = AR6Infiller.from_ar6_config(
        ar6_infilling_db_file=AR6_INFILLING_DB_FILE,
        ar6_infilling_db_cfcs_file=AR6_INFILLING_DB_CFCS_FILE,
        run_checks=False,
        n_processes=None,  # not parallel
        progress=False,
    )

    res = infiller(harmonised)

    exp = (
        get_ar6_infilled_emissions(
            model=model,
            scenario=scenario,
            processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
        )
        # Ignore aggregate stuff
        .loc[
            ~pix.ismatch(variable=["**CO2", "**Kyoto**", "**F-Gases", "**HFC", "**PFC"])
        ]
    )

    assert_frame_equal(res, exp)


@pytest.mark.slow
def test_key_testing_scenarios_all_at_once_parallel():
    harmonised_l = []
    exp_l = []
    for model, scenario in KEY_TESTING_MODEL_SCENARIOS:
        harmonised_l.append(
            get_ar6_harmonised_emissions(
                model=model,
                scenario=scenario,
                processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
            )
            # Ignore aggregate stuff
            # (but keep CO2 total, that is needed)
            .loc[~pix.ismatch(variable=["**Kyoto**", "**F-Gases", "**HFC", "**PFC"])]
        )
        exp_l.append(
            get_ar6_infilled_emissions(
                model=model,
                scenario=scenario,
                processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
            )
            # Ignore aggregate stuff
            .loc[
                ~pix.ismatch(
                    variable=["**CO2", "**Kyoto**", "**F-Gases", "**HFC", "**PFC"]
                )
            ]
        )

    harmonised = pd.concat(harmonised_l)
    exp = pd.concat(exp_l)

    infiller = AR6Infiller.from_ar6_config(
        ar6_infilling_db_file=AR6_INFILLING_DB_FILE,
        ar6_infilling_db_cfcs_file=AR6_INFILLING_DB_CFCS_FILE,
        run_checks=False,
        # n_processes=None,  # not parallel
        # progress=False,
    )

    res = infiller(harmonised)

    assert_frame_equal(res, exp)
