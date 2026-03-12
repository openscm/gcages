"""
Test harmonisation compared to notebooks used for CMIP7 ScenarioMIP

Note that you could use this to test all scenarios,
but we don't to save computational resources.
Instead, we just test the markers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.cmip7_scenariomip import (
    create_cmip7_scenariomip_country_harmoniser,
    create_cmip7_scenariomip_global_harmoniser,
)
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import (
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_cmip7_scenariomip_harmonised_emissions,
    get_cmip7_scenariomip_pre_processed_emissions,
    get_key_testing_model_scenario_parameters,
)

pix = pytest.importorskip("pandas_indexing")

CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE = (
    # Downloaded with
    # `tests/regression/cmip7-scenariomip/download_cmip7_scenariomip_history.py`
    Path(__file__).parents[0]
    / "cmip7-scenariomip-workflow-inputs"
    / "history_cmip7_scenariomip.csv"
)
CMIP7_SCENARIOMIP_HISTORICAL_COUNTRY_EMISSIONS_FILE = (
    # Downloaded with
    # `tests/regression/cmip7-scenariomip/download_cmip7_scenariomip_history.py`
    Path(__file__).parents[0]
    / "cmip7-scenariomip-workflow-inputs"
    / "history_gridding_cdr.csv"
    # / "country_history_cmip7_scenariomip.csv"
)
PROCESSED_CMIP7_SCENARIOMIP_OUTPUT_DIR = (
    Path(__file__).parents[0] / "cmip7-scenariomip-output"
)


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

    # Interpolate to annual values.
    # TODO: put in pre-processing so end to end tests don't fail.
    out_years = np.arange(pre_processed.columns.min(), pre_processed.columns.max() + 1)

    pre_processed_interpolated = (
        pre_processed.reindex(columns=out_years)
        .sort_index(axis="columns")
        .T.interpolate(method="index")
        .T
    )

    # Harmonise
    # Only works if aneris installed
    pytest.importorskip("aneris")
    harmoniser = create_cmip7_scenariomip_global_harmoniser(
        cmip7_scenariomip_global_historical_emissions_file=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        aneris_global_overrides_file=PROCESSED_CMIP7_SCENARIOMIP_OUTPUT_DIR
        / "aneris-overrides-global.csv",
        n_processes=None,  # not parallel
        progress=False,
    )
    res = harmoniser(pre_processed_interpolated)

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


@get_key_testing_model_scenario_parameters(
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS
)
def test_individual_scenario_country(model, scenario):
    pre_processed_all = get_cmip7_scenariomip_pre_processed_emissions(
        model=model,
        scenario=scenario,
        processed_cmip7_scenariomip_output_data_dir=PROCESSED_CMIP7_SCENARIOMIP_OUTPUT_DIR,
    )

    pre_processed = pre_processed_all.loc[
        pix.ismatch(stage="gridding_emissions")
    ].reset_index("stage", drop=True)
    if pre_processed.empty:
        raise AssertionError

    # Interpolate to annual values.
    # TODO: put in pre-processing so end to end tests don't fail.
    out_years = np.arange(pre_processed.columns.min(), pre_processed.columns.max() + 1)

    pre_processed_interpolated = (
        pre_processed.reindex(columns=out_years)
        .sort_index(axis="columns")
        .T.interpolate(method="index")
        .T
    )

    # Harmonise
    # Only works if aneris installed
    pytest.importorskip("aneris")
    harmoniser = create_cmip7_scenariomip_country_harmoniser(
        cmip7_scenariomip_country_historical_emissions_file=CMIP7_SCENARIOMIP_HISTORICAL_COUNTRY_EMISSIONS_FILE,
        aneris_country_overrides_file=PROCESSED_CMIP7_SCENARIOMIP_OUTPUT_DIR
        / "aneris-overrides-gridding.csv",
        n_processes=None,  # not parallel
        progress=False,
    )
    res = harmoniser(pre_processed_interpolated)

    # Get expected result
    harmonised_all = get_cmip7_scenariomip_harmonised_emissions(
        model=model,
        scenario=scenario,
        processed_cmip7_scenariomip_output_data_dir=PROCESSED_CMIP7_SCENARIOMIP_OUTPUT_DIR,
    )

    exp = harmonised_all.loc[pix.ismatch(workflow="gridding")].reset_index(
        "workflow", drop=True
    )
    if exp.empty:
        raise AssertionError

    # Convert names to gcages naming before comparing
    # exp = update_index_levels_func(
    #     exp,
    #     {
    #         "variable": lambda x: convert_variable_name(
    #             x,
    #             from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    #             to_convention=SupportedNamingConventions.GCAGES,
    #         )
    #     },
    #     copy=False,
    # )

    assert_frame_equal(res, exp)


@pytest.mark.slow
def test_key_testing_scenarios_all_at_once_parallel():
    pre_processed_l = []
    exp_l = []
    for model, scenario in KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS:
        pre_processed_l.append(
            get_cmip7_scenariomip_pre_processed_emissions(
                model=model,
                scenario=scenario,
                processed_cmip7_scenariomip_output_data_dir=PROCESSED_CMIP7_SCENARIOMIP_OUTPUT_DIR,
            )
        )
        exp_l.append(
            get_cmip7_scenariomip_harmonised_emissions(
                model=model,
                scenario=scenario,
                processed_cmip7_scenariomip_output_data_dir=PROCESSED_CMIP7_SCENARIOMIP_OUTPUT_DIR,
            )
        )

    pre_processed_all = pd.concat(pre_processed_l)
    pre_processed = pre_processed_all.loc[
        pix.ismatch(region="World", stage="global_workflow_emissions")
    ].reset_index("stage", drop=True)
    if pre_processed.empty:
        raise AssertionError

    # Interpolate to annual values.
    # TODO: put in pre-processing so end to end tests don't fail.
    out_years = np.arange(pre_processed.columns.min(), pre_processed.columns.max() + 1)

    pre_processed_interpolated = (
        pre_processed.reindex(columns=out_years)
        .sort_index(axis="columns")
        .T.interpolate(method="index")
        .T
    )

    exp = pd.concat(exp_l)

    # Harmonise
    # Only works if aneris installed
    pytest.importorskip("aneris")
    harmoniser = create_cmip7_scenariomip_global_harmoniser(
        cmip7_scenariomip_global_historical_emissions_file=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        aneris_global_overrides_file=PROCESSED_CMIP7_SCENARIOMIP_OUTPUT_DIR
        / "aneris-overrides-global.csv",
        run_checks=False,
        # run in parallel is the default
        # n_processes=None,
        # run with progress bars is the default
        # progress=False,
    )
    res = harmoniser(pre_processed_interpolated)

    exp = exp.loc[pix.ismatch(workflow="global")].reset_index("workflow", drop=True)
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
