"""
Test infilling compared for CMIP7 ScenarioMIP
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip.infilling import (
    create_cmip7_scenariomip_infilled_df,
)

# from gcages.completeness import get_missing_levels
# from gcages.index_manipulation import (
# create_levels_based_on_existing,
# set_new_single_value_levels,
# )
# from pandas_openscm.index_manipulation import update_index_levels_func
# from gcages.ar6 import AR6Harmoniser, AR6PreProcessor
# from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import (
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    #     get_ar6_harmonised_emissions,
    #     get_ar6_raw_emissions,
    get_key_testing_model_scenario_parameters,
)

pix = pytest.importorskip("pandas_indexing")

HARMONISED_CMIP7_DIR = Path(__file__).parents[0] / "cmip7-scenariomip-output"
AUX_INPUT_DIR = Path(__file__).parents[0] / "cmip7-scenariomip-workflow-inputs"
AUX_FILES = [
    "history_cmip7_scenariomip.csv",
    "infilling_cmip7_scenariomip.csv",
    "cmip7_ghg_inversions.csv",
]
INFILLED_CMIP7_SCENARIOMIP_OUT_DIR = (
    Path(__file__).parents[0] / "cmip7-scenariomip-output"
)

HARMONISATION_YEAR = 2023


@get_key_testing_model_scenario_parameters(
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS
)
def test_individual_scenario(model, scenario):
    # Loading harmonised results
    file = next(HARMONISED_CMIP7_DIR.glob(f"{model}_{scenario}_harmonised.csv"), None)
    with open(file) as f:
        harmonised_df = load_timeseries_csv(
            f,
            lower_column_names=True,
            index_columns=[
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "workflow",
            ],
            out_columns_type=int,
        )

        harmonised_df.columns.name = "year"
        harmonised_df = harmonised_df.loc[pix.ismatch(workflow="global")].reset_index(
            ["workflow"], drop=True
        )

    # Loading harmonised results
    file = next(INFILLED_CMIP7_SCENARIOMIP_OUT_DIR.glob(f"infilled_{model}.csv"), None)
    with open(file) as f:
        exp = load_timeseries_csv(
            f,
            lower_column_names=True,
            index_columns=["model", "scenario", "region", "variable", "unit"],
            out_columns_type=int,
        )
        exp = exp.loc[:, HARMONISATION_YEAR:2100]
        exp.columns.name = "year"
        # Select scenario and drop aggregated/cumulative rows
        exp = exp.loc[
            pix.ismatch(scenario=scenario)
            & ~pix.ismatch(variable=["**Kyoto**", "Cumulative**", "**CO2", "**GHG**"])
        ]

    infilled = create_cmip7_scenariomip_infilled_df(
        harmonised_df,
        cmip7_scenariomip_global_historical_emissions_file=AUX_INPUT_DIR.joinpath(
            AUX_FILES[0]
        ),
        cmip7_scenariomip_infilling_leader_emissions_file=AUX_INPUT_DIR.joinpath(
            AUX_FILES[1]
        ),
        cmip7_ghg_inversions_file=AUX_INPUT_DIR.joinpath(AUX_FILES[2]),
        ur=None,
    )

    assert_frame_equal(infilled.complete, exp)
