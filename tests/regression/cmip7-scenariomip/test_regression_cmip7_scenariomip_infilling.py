"""
Test infilling compared to CMIP7 ScenarioMIP
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gcages.cmip7_scenariomip.infilling import (
    CMIP7ScenarioMIPInfiller,
)
from gcages.testing import (
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_cmip7_scenariomip_harmonised_emissions,
    get_cmip7_scenariomip_infilled_emissions,
    get_key_testing_model_scenario_parameters,
)

pix = pytest.importorskip("pandas_indexing")

OUTPUT_CMIP7_DIR = Path(__file__).parents[0] / "cmip7-scenariomip-output"
INPUT_DIR = Path(__file__).parents[0] / "cmip7-scenariomip-workflow-inputs"

HARMONISATION_YEAR = 2023


# @get_key_testing_model_scenario_parameters(
#     KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS
# )
# def test_individual_scenario(model, scenario):
#     # Load harmonised results
#     harmonised_df = get_cmip7_scenariomip_harmonised_emissions(
#         model=model,
#         scenario=scenario,
#         processed_cmip7_scenariomip_output_data_dir=OUTPUT_CMIP7_DIR,
#     )
#
#     harmonised_df = harmonised_df.loc[pix.ismatch(workflow="for_scms")].reset_index(
#         ["workflow"], drop=True
#     )
#
#     # Load infilled results
#     exp = get_cmip7_scenariomip_infilled_emissions(
#         model=model,
#         scenario=scenario,
#         processed_cmip7_scenariomip_output_data_dir=OUTPUT_CMIP7_DIR,
#     )
#
#     infilled = create_cmip7_scenariomip_infilled_df(
#         harmonised_df,
#         cmip7_scenariomip_global_historical_emissions_file=INPUT_DIR.joinpath(
#             "history_cmip7_scenariomip.csv"
#         ),
#         cmip7_scenariomip_infilling_leader_emissions_file=INPUT_DIR.joinpath(
#             "infilling_db_cmip7_scenariomip.csv"
#         ),
#         cmip7_ghg_inversions_file=INPUT_DIR.joinpath("cmip7_ghg_inversions.csv"),
#         ur=None,
#     )
#
#     assert_frame_equal(infilled.complete, exp)
#


@get_key_testing_model_scenario_parameters(
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS
)
def test_individual_scenario_class(model, scenario):
    # Load harmonised results
    harmonised_df = get_cmip7_scenariomip_harmonised_emissions(
        model=model,
        scenario=scenario,
        processed_cmip7_scenariomip_output_data_dir=OUTPUT_CMIP7_DIR,
    )

    harmonised_df = harmonised_df.loc[pix.ismatch(workflow="for_scms")].reset_index(
        ["workflow"], drop=True
    )

    # Load infilled results
    exp = get_cmip7_scenariomip_infilled_emissions(
        model=model,
        scenario=scenario,
        processed_cmip7_scenariomip_output_data_dir=OUTPUT_CMIP7_DIR,
    )

    infiller = CMIP7ScenarioMIPInfiller.from_cmip7_scenariomip_config(
        cmip7_scenariomip_infilling_leader_emissions_file=INPUT_DIR.joinpath(
            "infilling_db_cmip7_scenariomip.csv"
        ),
        cmip7_ghg_inversions_file=INPUT_DIR.joinpath("cmip7_ghg_inversions.csv"),
        cmip7_scenariomip_global_historical_emissions_file=INPUT_DIR.joinpath(
            "history_cmip7_scenariomip.csv"
        ),
        ur=None,
    )
    infilled = infiller(harmonised_df)

    assert_frame_equal(infilled, exp)
