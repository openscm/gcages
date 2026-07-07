"""
Test simple climate model running compared to SCI June 2026 run
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pandas as pd
import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.sci_june_2026.post_processing import SCIJune2026PostProcessor
from gcages.sci_june_2026.scm_running import SCIJune2026SCMRunner
from gcages.testing import (
    KEY_SCI_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_key_testing_model_scenario_parameters,
    guess_magicc_exe,
)

pix = pytest.importorskip("pandas_indexing")

CMIP7_SCENARIOMIP_OUT_DIR = Path(__file__).parents[0] / "cmip7-scenariomip-output"

# Paths
SCI_INPUT_DIR = Path(__file__).parents[0] / "sci_workflow_inputs"
SCI_OUTPUT_DIR = Path(__file__).parents[0] / "sci_workflow_expected_outputs"

PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR = (
    Path(__file__).parents[1] / "cmip7-scenariomip/cmip7-scenariomip-workflow-inputs"
)
CMIP7_SCENARIOMIP_MAGICC_EXECUTABLES_DIR = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "magicc-v7.6.0a3/bin"
)
CMIP7_SCENARIOMIP_MAGICC_PROBABILISTIC_CONFIG_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
    / "magicc-v7.6.0a3/configs/magicc-ar7-fast-track-drawnset-v0-3-0.json"
)
CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "history_cmip7_scenariomip.csv"
)

HARMONISATION_YEAR = 2023


@pytest.mark.skip_ci_default
@pytest.mark.slow
@pytest.mark.magicc_v760a3
@get_key_testing_model_scenario_parameters(KEY_SCI_TESTING_MODEL_SCENARIOS)
def test_individual_scenario(model, scenario, monkeypatch):
    # Loading infilled results
    file = (
        Path(__file__).parents[0] / "sci_workflow_expected_outputs" / "sci_complete.csv"
    )
    input_df = load_timeseries_csv(
        file,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )
    complete = input_df.loc[pix.ismatch(model=model, scenario=scenario)]
    # Select scenario and drop aggregated/cumulative rows
    # complete = update_index_levels_func(
    #     complete,
    #     {
    #         "variable": partial(
    #             convert_variable_name,
    #             from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    #             to_convention=SupportedNamingConventions.GCAGES,
    #         )
    #     },
    # )
    #

    monkeypatch.delenv("MAGICC_EXECUTABLE_7", raising=False)
    scm_runner = SCIJune2026SCMRunner.from_files(
        magicc_exe_path=guess_magicc_exe(CMIP7_SCENARIOMIP_MAGICC_EXECUTABLES_DIR),
        magicc_prob_distribution_path=CMIP7_SCENARIOMIP_MAGICC_PROBABILISTIC_CONFIG_FILE,
        output_variables=("Surface Air Temperature Change",),
        historical_emissions_path=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        harmonisation_year=HARMONISATION_YEAR,
        n_processes=multiprocessing.cpu_count(),
        batch_size_scenarios=15,
    )
    n_cfgs = 6
    scm_runner.climate_models_cfgs["MAGICC7"] = scm_runner.climate_models_cfgs[
        "MAGICC7"
    ][:n_cfgs]

    scm_results = scm_runner(complete)

    # Post-processing
    post_processor = SCIJune2026PostProcessor.from_cmip7_scenariomip_config()
    post_processed = post_processor(scm_results)

    # Loading and assessing quantiles timeseries results
    file = SCI_OUTPUT_DIR / "sci_post-processed-timeseries_quantile.csv"
    exp_quantiles = load_timeseries_csv(
        file,
        lower_column_names=True,
        index_columns=[
            "climate_model",
            "model",
            "region",
            "scenario",
            "unit",
            "variable",
            "quantile",
        ],
        out_columns_type=int,
        out_columns_name="time",
    )
    exp_quantiles = exp_quantiles.loc[pix.ismatch(model=model, scenario=scenario)]

    exp_quantiles.index = exp_quantiles.index.set_levels(
        exp_quantiles.index.levels[exp_quantiles.index.names.index("quantile")].round(
            4
        ),
        level="quantile",
    )
    processed_quantiles = post_processed.timeseries_quantile
    processed_quantiles.index = processed_quantiles.index.set_levels(
        exp_quantiles.index.levels[exp_quantiles.index.names.index("quantile")].round(
            4
        ),
        level="quantile",
    )

    assert_frame_equal(
        processed_quantiles,
        exp_quantiles,
        rtol=1e-7,
    )

    # Loading and categories
    file = SCI_OUTPUT_DIR / "sci_post-processed-metadata_categories.csv"
    exp_categories = pd.read_csv(file)
    exp_categories = exp_categories.set_index(
        ["model", "scenario", "climate_model", "metric", "0"]
    )
    exp_categories = exp_categories.loc[pix.ismatch(model=model, scenario=scenario)]

    assert (
        post_processed.metadata_categories.values[0]
        == exp_categories.index.get_level_values("0")[0]
    )
    assert (
        post_processed.metadata_categories.values[1]
        == exp_categories.index.get_level_values("0")[1]
    )
