"""
Test infilling compared for CMIP7 ScenarioMIP
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip.harmonisation import (
    load_cmip7_scenariomip_historical_emissions,
)
from gcages.cmip7_scenariomip.scm_running import (
    CMIP7_SCENARIOMIP_SCMRunner,
)
from gcages.testing import (
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_key_testing_model_scenario_parameters,
)

pix = pytest.importorskip("pandas_indexing")

CMIP7_SCENARIOMIP_OUT_DIR = Path(__file__).parents[0] / "cmip7-scenariomip-output"

PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR = (
    Path(__file__).parents[0] / "cmip7-scenariomip-workflow-inputs"
)
CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "history_cmip7_scenariomip.csv"
)

MAGIC_EXE = PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "magicc-v7.6.0a3/bin/magicc"
MAGICC_CMIP7_PROBABILISTIC_CONFIG_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
    / "magicc-v7.6.0a3/configs/magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

HARMONISATION_YEAR = 2023


@pytest.mark.skip_ci_default
@pytest.mark.slow
@get_key_testing_model_scenario_parameters(
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS
)
def test_individual_scenario(model, scenario):
    # Load history
    historical_emissions = load_cmip7_scenariomip_historical_emissions(
        filepath=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        check_hash=True,
    )

    # Loading infilled results
    file = CMIP7_SCENARIOMIP_OUT_DIR / f"{model}_{scenario}_infilled.csv"
    infilled = load_timeseries_csv(
        file,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )
    # Select scenario and drop aggregated/cumulative rows
    infilled = infilled.loc[
        pix.ismatch(scenario=scenario)
        & ~pix.ismatch(variable=["**Kyoto**", "Cumulative**", "**CO2", "**GHG**"])
    ]

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

    scm_runner = CMIP7_SCENARIOMIP_SCMRunner.from_cmip7_scenariomip_config(
        magicc_exe_path=MAGIC_EXE,
        magicc_prob_distribution_path=MAGICC_CMIP7_PROBABILISTIC_CONFIG_FILE,
        output_variables=("Surface Air Temperature Change",),
        historical_emissions=historical_emissions,
        harmonisation_year=HARMONISATION_YEAR,
        n_processes=multiprocessing.cpu_count(),
    )

    scm_results = scm_runner(infilled)

    assert_frame_equal(
        scm_results[
            scm_results.index.get_level_values("variable").str.contains(
                "Surface Air Temperature Change"
            )
        ],
        exp_temperature,
        rtol=1e-1,
    )
