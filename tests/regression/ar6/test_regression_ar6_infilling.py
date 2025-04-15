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
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.ar6 import AR6Infiller, get_ar6_full_historical_emissions
from gcages.renaming import SupportedNamingConventions, convert_variable_name
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


def strip_off_ar6_harmonised_prefix_and_convert_to_gcages(
    indf: pd.DataFrame,
) -> pd.DataFrame:
    indf = update_index_levels_func(
        indf,
        {
            "variable": lambda x: convert_variable_name(
                x.replace("AR6 climate diagnostics|Harmonized|", ""),
                from_convention=SupportedNamingConventions.IAMC,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
        copy=False,
    )

    return indf


def add_ar6_infilled_prefix_and_convert_to_iamc_and_add_harmonised(
    indf: pd.DataFrame,
    harmonised: pd.DataFrame,
) -> pd.DataFrame:
    res = update_index_levels_func(
        # AR6 put harmonised in the infilled group too for some reason,
        # except CO2
        pd.concat(
            [
                indf,
                harmonised.loc[
                    ~harmonised.index.get_level_values("variable").str.endswith("CO2")
                ],
            ]
        ),
        {
            "variable": lambda x: (
                "AR6 climate diagnostics|Infilled|"
                + convert_variable_name(
                    x,
                    from_convention=SupportedNamingConventions.GCAGES,
                    to_convention=SupportedNamingConventions.IAMC,
                )
            ),
            "unit": lambda x: x.replace("HFC245fa", "HFC245ca").replace(
                "HFC4310", "HFC43-10"
            ),
        },
    )

    return res


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

    harmonised = strip_off_ar6_harmonised_prefix_and_convert_to_gcages(harmonised)

    infiller = AR6Infiller.from_ar6_config(
        ar6_infilling_db_file=AR6_INFILLING_DB_FILE,
        ar6_infilling_db_cfcs_file=AR6_INFILLING_DB_CFCS_FILE,
        historical_emissions=get_ar6_full_historical_emissions(
            AR6_INFILLING_DB_CFCS_FILE
        ),
        harmonisation_year=2015,
        progress=False,
        n_processes=None,  # not parallel
    )
    assert not infiller.progress

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

    res_comparable = add_ar6_infilled_prefix_and_convert_to_iamc_and_add_harmonised(
        res, harmonised=harmonised
    )
    assert_frame_equal(res_comparable, exp)


@pytest.mark.slow
def test_key_testing_scenarios_all_at_once_parallel():
    # Required for progress bars
    pytest.importorskip("tqdm.auto")

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

    harmonised = strip_off_ar6_harmonised_prefix_and_convert_to_gcages(
        pd.concat(harmonised_l)
    )
    exp = pd.concat(exp_l)

    infiller = AR6Infiller.from_ar6_config(
        ar6_infilling_db_file=AR6_INFILLING_DB_FILE,
        ar6_infilling_db_cfcs_file=AR6_INFILLING_DB_CFCS_FILE,
        # run in parallel is the default
        # n_processes=None,
        # run with progress bars is the default
        # progress=False,
        historical_emissions=get_ar6_full_historical_emissions(
            AR6_INFILLING_DB_CFCS_FILE
        ),
        harmonisation_year=2015,
    )

    res = infiller(harmonised)

    res_comparable = add_ar6_infilled_prefix_and_convert_to_iamc_and_add_harmonised(
        res, harmonised=harmonised
    )
    assert_frame_equal(res_comparable, exp)
