"""
Test infilling compared for CMIP7 ScenarioMIP
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip.harmonisation import (
    load_cmip7_scenariomip_global_historical_emissions,
)
from gcages.cmip7_scenariomip.scm_running_aux import (
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
    historical_emissions = load_cmip7_scenariomip_global_historical_emissions(
        filepath=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        check_hash=True,
    )

    # Loading infilled results
    file = next(CMIP7_SCENARIOMIP_OUT_DIR.glob(f"infilled_{model}.csv"), None)
    with open(file) as f:
        infilled = load_timeseries_csv(
            f,
            lower_column_names=True,
            index_columns=["model", "scenario", "region", "variable", "unit"],
            out_columns_type=int,
        )
        # exp = exp.loc[:, HARMONISATION_YEAR:2100]
        infilled.columns.name = "year"
        # Select scenario and drop aggregated/cumulative rows
        infilled = infilled.loc[
            pix.ismatch(scenario=scenario)
            & ~pix.ismatch(variable=["**Kyoto**", "Cumulative**", "**CO2", "**GHG**"])
        ]

    # Loading expected results
    file = next(CMIP7_SCENARIOMIP_OUT_DIR.glob(f"{model}_{scenario}_GSAT.csv"), None)
    with open(file) as f:
        exp_temperature = load_timeseries_csv(
            f,
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
        # exp = exp.loc[:, HARMONISATION_YEAR:2100]
        exp_temperature.columns.name = "time"

    scm_runner = CMIP7_SCENARIOMIP_SCMRunner.from_cmip7_scenariomip_config(
        magicc_exe_path=MAGIC_EXE,
        magicc_prob_distribution_path=MAGICC_CMIP7_PROBABILISTIC_CONFIG_FILE,
        output_variables=("Surface Air Temperature Change",),
        historical_emissions=historical_emissions,
        harmonisation_year=2023,
        n_processes=multiprocessing.cpu_count(),
    )

    # post_processor = AR6PostProcessor.from_ar6_config(n_processes=None)

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

    # post_processed = post_processor(scm_results)

    # res_temperature_percentiles_comparable = convert_to_ar6_percentile_output(
    #     post_processed.timeseries_quantile.loc[
    #         pix.ismatch(variable="Surface Temperature (GSAT)")
    #     ]
    # )

    # exp_temperature_percentiles = get_ar6_temperature_outputs(
    #     model=model,
    #     scenario=scenario,
    #     processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
    # )

    # assert_frame_equal(
    #     res_temperature_percentiles_comparable.loc[
    #     :, exp_temperature_percentiles.columns
    #     ],
    #     exp_temperature_percentiles,
    #     rtol=1e-5,
    # )

    # metadata_compare_cols = [
    #     "Category",
    #     "Category_name",
    #     "Median peak warming (MAGICCv7.5.3)",
    #     "P33 peak warming (MAGICCv7.5.3)",
    #     "Median warming in 2100 (MAGICCv7.5.3)",
    #     "Median year of peak warming (MAGICCv7.5.3)",
    #     "Exceedance Probability 1.5C (MAGICCv7.5.3)",
    #     "Exceedance Probability 2.0C (MAGICCv7.5.3)",
    # ]
    # exp_numerical_cols = list(
    #     set(metadata_compare_cols) - {"Category", "Category_name"}
    # )
    # exp_metadata[exp_numerical_cols] = exp_metadata[exp_numerical_cols].astype(float)
    #
    # post_processed_metadata_comparable = get_post_processed_metadata_comparable(
    #     post_processed
    # )
    # # If needed, use the failed vetting flag
    # post_processed_metadata_comparable.loc[
    #     exp_metadata["Category"] == "failed-vetting",
    #     ["Category", "Category_name"],
    # ] = "failed-vetting"
    # pd.testing.assert_frame_equal(
    #     post_processed_metadata_comparable[metadata_compare_cols],
    #     exp_metadata[metadata_compare_cols],
    #     rtol=1e-5,
    # )
