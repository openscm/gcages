"""
Test harmonisation compared to notebooks used for CMIP7 ScenarioMIP

Note that you could use this to test all scenarios,
but we don't to save computational resources.
Instead, we just test the markers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.cmip7_scenariomip import CMIP7ScenarioMIPHarmoniser
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import (
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_cmip7_scenariomip_harmonised_emissions,
    get_cmip7_scenariomip_pre_processed_emissions,
    get_key_testing_model_scenario_parameters,
)

pix = pytest.importorskip("pandas_indexing")

CMIP7_SCENARIOMIP_HISTORICAL_EMISSIONS_FILE = (
    # Downloaded with
    # `tests/regression/cmip7-scenariomip/download_cmip7_scenariomip_history.py`
    Path(__file__).parents[0]
    / "cmip7-scenariomip-workflow-inputs"
    / "history_cmip7_scenariomip.csv"
)
PROCESSED_CMIP7_SCENARIOMIP_OUTPUT_DIR = (
    Path(__file__).parents[0] / "cmip7-scenariomip-output"
)


@pytest.mark.skip_ci_default
@get_key_testing_model_scenario_parameters(
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS
)
def test_individual_scenario_global(model, scenario):
    pre_processed_all = get_cmip7_scenariomip_pre_processed_emissions(
        model=model,
        scenario=scenario,
        processed_cmip7_scenariomip_output_data_dir=PROCESSED_CMIP7_SCENARIOMIP_OUTPUT_DIR,
    )

    pre_processed = pre_processed_all.loc[
        pix.ismatch(region="World", stage="global_workflow_emissions")
    ].reset_index("stage", drop=True)
    if pre_processed.empty:
        raise AssertionError

    # Harmonise
    # Only works if aneris installed
    pytest.importorskip("aneris")
    harmoniser = CMIP7ScenarioMIPHarmoniser.from_cmip7_scenariomip_config(
        cmip7_scenariomip_historical_emissions_file=CMIP7_SCENARIOMIP_HISTORICAL_EMISSIONS_FILE,
        harmonisation_year=2023,
        n_processes=None,  # not parallel
        progress=False,
    )
    res = harmoniser(pre_processed)

    # Get expected result
    harmonised_all = get_cmip7_scenariomip_harmonised_emissions(
        model=model,
        scenario=scenario,
        processed_cmip7_scenariomip_output_data_dir=PROCESSED_CMIP7_SCENARIOMIP_OUTPUT_DIR,
    )

    exp = harmonised_all.loc[pix.ismatch(workflow="global")].reset_index(
        "workflow", drop=True
    )
    if exp.empty:
        raise AssertionError

    # Convert names to gcages naming before comparing
    exp = update_index_levels_func(
        exp,
        {
            "variable": lambda x: convert_variable_name(
                x,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
        copy=False,
    )

    assert_frame_equal(res, exp)


# @pytest.mark.slow
# @pytest.mark.skip_ci_default
# def test_key_testing_scenarios_all_at_once_parallel():
#     raw_l = []
#     exp_l = []
#     for model, scenario in KEY_AR6_TESTING_MODEL_SCENARIOS:
#         raw_l.append(
#             get_ar6_raw_emissions(
#                 model=model,
#                 scenario=scenario,
#                 processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
#             )
#         )
#         exp_l.append(
#             get_ar6_harmonised_emissions(
#                 model=model,
#                 scenario=scenario,
#                 processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
#             )
#             # Ignore aggregate stuff
#             .loc[~pix.ismatch(variable=["**Kyoto**", "**F-Gases", "**HFC", "**PFC"])]
#         )
#
#     raw = strip_off_ar6_prefix(pd.concat(raw_l))
#     exp = pd.concat(exp_l)
#
#     pre_processor = AR6PreProcessor.from_ar6_config(
#         # run in parallel is the default
#         # n_processes=None,
#         # run with progress bars is the default
#         # progress=False,
#     )
#
#     # Only works if aneris installed
#     pytest.importorskip("aneris")
#     harmoniser = AR6Harmoniser.from_ar6_config(
#         ar6_historical_emissions_file=AR6_HISTORICAL_EMISSIONS_FILE,
#         run_checks=False,
#         # run in parallel is the default
#         # n_processes=None,
#         # run with progress bars is the default
#         # progress=False,
#     )
#
#     pre_processed = pre_processor(raw)
#     res = harmoniser(pre_processed)
#     res_comparable = add_ar6_prefix_and_convert_to_iamc(res)
#
#     assert_frame_equal(res_comparable, exp)
