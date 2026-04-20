"""
Test simple climate model running compared for CMIP7 ScenarioMIP
"""

from __future__ import annotations

import multiprocessing
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip.post_processing import CMIP7ScenarioMIPPostProcessor
from gcages.cmip7_scenariomip.scm_running import (
    CMIP7ScenarioMIPSCMRunner,
)
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import (
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_key_testing_model_scenario_parameters,
    guess_magicc_exe,
)

pix = pytest.importorskip("pandas_indexing")

CMIP7_SCENARIOMIP_OUT_DIR = Path(__file__).parents[0] / "cmip7-scenariomip-output"

PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR = (
    Path(__file__).parents[0] / "cmip7-scenariomip-workflow-inputs"
)
CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "history_cmip7_scenariomip.csv"
)

CMIP7_SCENARIOMIP_MAGICC_EXECUTABLES_DIR = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "magicc-v7.6.0a3/bin"
)
CMIP7_SCENARIOMIP_MAGICC_PROBABILISTIC_CONFIG_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
    / "magicc-v7.6.0a3/configs/magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

HARMONISATION_YEAR = 2023


@pytest.mark.skip_ci_default
@pytest.mark.slow
@pytest.mark.magicc_v760a3
@get_key_testing_model_scenario_parameters(
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS
)
def test_individual_scenario(model, scenario):
    # Loading infilled results
    file = CMIP7_SCENARIOMIP_OUT_DIR / f"{model}_{scenario}_complete.csv"
    complete = load_timeseries_csv(
        file,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )
    # Select scenario and drop aggregated/cumulative rows
    complete = complete.loc[
        pix.ismatch(scenario=scenario)
        & ~pix.ismatch(variable=["**Kyoto**", "Cumulative**", "**CO2", "**GHG**"])
    ]
    complete = update_index_levels_func(
        complete,
        {
            "variable": partial(
                convert_variable_name,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
    )

    # Loading expected results
    file = CMIP7_SCENARIOMIP_OUT_DIR / f"{model}_{scenario}_GSAT.csv"
    exp_temperature = load_timeseries_csv(
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
    )
    exp_temperature.columns.name = "time"

    scm_runner = CMIP7ScenarioMIPSCMRunner.from_cmip7_scenariomip_config(
        magicc_exe_path=guess_magicc_exe(CMIP7_SCENARIOMIP_MAGICC_EXECUTABLES_DIR),
        magicc_prob_distribution_path=CMIP7_SCENARIOMIP_MAGICC_PROBABILISTIC_CONFIG_FILE,
        output_variables=("Surface Air Temperature Change",),
        historical_emissions_path=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        harmonisation_year=HARMONISATION_YEAR,
        n_processes=multiprocessing.cpu_count(),
    )

    scm_results = scm_runner(complete)

    assert_frame_equal(
        scm_results[
            scm_results.index.get_level_values("variable").str.contains(
                "Surface Air Temperature Change"
            )
        ].iloc[:10],
        exp_temperature,
    )

    # Post-processing
    post_processor = CMIP7ScenarioMIPPostProcessor.from_cmip7_scenariomip_config()
    post_processed = post_processor(scm_results)

    # Loading and assessing quantiles timeseries results
    file = (
        CMIP7_SCENARIOMIP_OUT_DIR / f"assessed-warming-timeseries-quantiles_{model}.csv"
    )
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
    exp_quantiles.index = exp_quantiles.index.set_levels(
        exp_quantiles.index.levels[exp_quantiles.index.names.index("quantile")].round(
            4
        ),
        level="quantile",
    )
    processed_quantiles = post_processed.timeseries_quantile.iloc[:, 250:]
    processed_quantiles.index = processed_quantiles.index.set_levels(
        exp_quantiles.index.levels[exp_quantiles.index.names.index("quantile")].round(
            4
        ),
        level="quantile",
    )

    assert_frame_equal(
        processed_quantiles,
        exp_quantiles,
        rtol=1e-8,
    )

    # Loading and categories
    file = CMIP7_SCENARIOMIP_OUT_DIR / f"categories_{model}.csv"
    exp_categories = pd.read_csv(file)

    assert post_processed.metadata_categories.values[0] == exp_categories["value.1"][2]
    assert post_processed.metadata_categories.values[1] == exp_categories["value"][2]
