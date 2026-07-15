"""
Test SCIjune2026 workflow
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pandas as pd
import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.sci_june_2026.harmonisation import (
    create_scijune2026_harmoniser,
)
from gcages.sci_june_2026.infilling import create_scijune2026_infiller
from gcages.sci_june_2026.post_processing import SCIJune2026PostProcessor
from gcages.sci_june_2026.pre_processing import SCIJune2026PreProcessor
from gcages.sci_june_2026.scm_running import SCIJune2026SCMRunner
from gcages.testing import (
    KEY_SCI_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_key_testing_model_scenario_parameters,
    guess_magicc_exe,
)

pix = pytest.importorskip("pandas_indexing")

# Paths
SCI_INPUT_DIR = Path(__file__).parents[0] / "sci_workflow_inputs"
SCI_OUTPUT_DIR = Path(__file__).parents[0] / "sci_workflow_expected_outputs"

PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR = (
    Path(__file__).parents[1] / "cmip7-scenariomip/cmip7-scenariomip-workflow-inputs"
)
SCI_HISTORICAL_EMISSIONS_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "history_cmip7_scenariomip.csv"
)
MAGICC_EXECUTABLES_DIR = PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "magicc-v7.6.0a3/bin"
MAGICC_PROBABILISTIC_CONFIG_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
    / "magicc-v7.6.0a3/configs/magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

# Variables
HARMONISATION_YEAR = 2023


@pytest.mark.skip_ci_default
@pytest.mark.slow
@get_key_testing_model_scenario_parameters(KEY_SCI_TESTING_MODEL_SCENARIOS)
def test_whole_pipeline(model, scenario, monkeypatch):
    """Test a few scenarios, not all to save compute time"""
    # LOADING SCENARIO
    file = SCI_INPUT_DIR / "SCI-2026-June-unique-testing-pathways.csv"

    input_df = pd.read_csv(file)

    input_df.columns = input_df.columns.str.lower()
    input_df = input_df.set_index(["model", "scenario", "region", "variable", "unit"])
    emissions = input_df.loc[
        pix.ismatch(
            model=model, scenario=scenario, variable="Emissions**", region="World"
        )
    ]
    emissions.columns = emissions.columns.astype(int)

    emissions = emissions.sort_index(axis="columns")
    pre_processor = SCIJune2026PreProcessor.from_sci_june2026_config(
        n_processes=None,  # run serially
        progress=False,
        run_checks=True,
    )
    pre_processed = pre_processor(emissions)

    if pre_processed.empty:
        raise AssertionError

    # Loading and assessing pre_processed results
    file = SCI_OUTPUT_DIR / "sci_pre-processed.csv"
    exp = load_timeseries_csv(
        file,
        lower_column_names=True,
        index_columns=[
            "model",
            "scenario",
            "region",
            "variable",
            "unit",
        ],
        out_columns_type=int,
    )
    exp = exp.loc[pix.ismatch(model=model, scenario=scenario)]

    assert_frame_equal(
        pre_processed.loc[:, 2023:],
        exp,
        rtol=1e-6,
    )

    # HARMONISATION
    # Only works if aneris installed
    pytest.importorskip("aneris")
    harmoniser = create_scijune2026_harmoniser(
        historical_emissions_file=SCI_HISTORICAL_EMISSIONS_FILE,
        aneris_overrides_file=SCI_INPUT_DIR / "sci_overrides.csv",
        harmonisation_year=HARMONISATION_YEAR,
    )
    harmonised = harmoniser(pre_processed)

    # Loading and assessing harmonised results
    file = SCI_OUTPUT_DIR / "sci_harmonised.csv"
    exp = load_timeseries_csv(
        file,
        lower_column_names=True,
        index_columns=[
            "model",
            "scenario",
            "region",
            "variable",
            "unit",
        ],
        out_columns_type=int,
    )
    exp = exp.loc[pix.ismatch(model=model, scenario=scenario)]

    assert_frame_equal(
        harmonised,
        exp,
        rtol=1e-6,
    )

    # INFILLING
    infiller = create_scijune2026_infiller(
        infilling_leader_emissions_file=SCI_INPUT_DIR / "infilling_db_sci.csv",
        ghg_inversions_file=PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
        / "cmip7_ghg_inversions.csv",
        historical_emissions_file=SCI_HISTORICAL_EMISSIONS_FILE,
        harmonisation_year=HARMONISATION_YEAR,
        pre_industrial_year=1750,
        run_checks=True,
    )
    complete = infiller(harmonised)

    # Loading and assessing infilled results
    file = SCI_OUTPUT_DIR / "sci_complete.csv"
    exp = load_timeseries_csv(
        file,
        lower_column_names=True,
        index_columns=[
            "model",
            "scenario",
            "region",
            "variable",
            "unit",
        ],
        out_columns_type=int,
        out_columns_name="year",
    )
    exp = exp.loc[pix.ismatch(model=model, scenario=scenario)]
    assert_frame_equal(
        complete,
        exp,
        rtol=1e-6,
    )

    # MAGICC and post_processing
    monkeypatch.delenv("MAGICC_EXECUTABLE_7", raising=False)
    scm_runner = SCIJune2026SCMRunner.from_cmip7_scenariomip_config(
        magicc_exe_path=guess_magicc_exe(MAGICC_EXECUTABLES_DIR),
        magicc_prob_distribution_path=MAGICC_PROBABILISTIC_CONFIG_FILE,
        output_variables=("Surface Air Temperature Change",),
        historical_emissions_path=SCI_HISTORICAL_EMISSIONS_FILE,
        harmonisation_year=HARMONISATION_YEAR,
        n_processes=multiprocessing.cpu_count(),
        batch_size_scenarios=15,
    )

    scm_results = scm_runner(complete)
    # Loading and assessing scm timeseries results
    file = SCI_OUTPUT_DIR / "scm_results.csv"
    exp = load_timeseries_csv(
        file,
        lower_column_names=True,
        index_columns=[
            "climate_model",
            "model",
            "region",
            "run_id",
            "scenario",
            "unit",
            "variable",
        ],
        out_columns_type=int,
        out_columns_name="time",
    )
    exp = exp.loc[pix.ismatch(model=model, scenario=scenario)]

    assert_frame_equal(
        scm_results[scm_results.index.get_level_values("run_id") == 599],
        exp,
        rtol=1e-4,
    )

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
        rtol=1e-6,
    )

    # Loading and categories
    file = SCI_OUTPUT_DIR / "sci_post-processed-metadata_categories.csv"
    exp_categories = pd.read_csv(file)
    exp_categories = exp_categories.set_index(
        ["climate_model", "model", "scenario", "metric"]
    )

    exp_categories = exp_categories.loc[pix.ismatch(model=model, scenario=scenario)]

    pd.testing.assert_frame_equal(
        post_processed.metadata_categories.to_frame("0"),
        exp_categories,
        check_like=True,
    )
