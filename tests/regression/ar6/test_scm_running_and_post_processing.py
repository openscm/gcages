"""
Test SCM running and post-processing compared to AR6

Note that you could use this to test all scenarios,
but we don't to save computational resources.
Instead, we just test a selection of scenarios
that cover the key paths from AR6.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pandas as pd
import pytest
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB

from gcages.ar6 import AR6PostProcessor, AR6SCMRunner
from gcages.testing import (
    KEY_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_ar6_infilled_emissions,
    get_ar6_metadata_outputs,
    get_ar6_temperature_outputs,
    get_key_testing_model_scenario_parameters,
    get_magicc_exe_path,
)

pix = pytest.importorskip("pandas_indexing")

AR6_MAGICC_PROBABILISTIC_CONFIG_FILE = (
    Path(__file__).parents[0]
    / "ar6-workflow-inputs"
    / "magicc-ar6-0fd0f62-f023edb-drawnset"
    / "0fd0f62-derived-metrics-id-f023edb-drawnset.json"
)
AR6_OUTPUT_DIR = Path(__file__).parents[0] / "ar6-output"
PROCESSED_AR6_DB_DIR = Path(__file__).parents[0] / "ar6-output-processed"


@pytest.mark.slow
@get_key_testing_model_scenario_parameters()
def test_individual_scenario(model, scenario):
    exp_metadata = get_ar6_metadata_outputs(
        model=model,
        scenario=scenario,
        ar6_output_data_dir=AR6_OUTPUT_DIR,
    )
    if (
        exp_metadata["Median peak warming (MAGICCv7.5.3)"] == "no-climate-assessment"
    ).all():
        pytest.skip(f"No climate assessment in AR6 for {model} {scenario}")

    infilled = (
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
    if infilled.empty:
        msg = f"No harmonised data for {model=} {scenario=}?"
        raise AssertionError(msg)

    magicc_exe = get_magicc_exe_path()
    scm_runner = AR6SCMRunner.from_ar6_config(
        run_checks=False,
        n_processes=multiprocessing.cpu_count(),
        progress=False,
        magicc_exe_path=magicc_exe,
        magicc_prob_distribution_path=AR6_MAGICC_PROBABILISTIC_CONFIG_FILE,
    )
    post_processor = AR6PostProcessor.from_ar6_config(
        run_checks=False, n_processes=None
    )

    scm_results = scm_runner(infilled)
    post_processed_timeseries, post_processed_metadata = post_processor(scm_results)

    res_temperature_percentiles = post_processed_timeseries.loc[
        pix.ismatch(
            variable="AR6 climate diagnostics|Surface Temperature (GSAT)|*|*Percentile"
        )
    ]

    exp_temperature_percentiles = get_ar6_temperature_outputs(
        model=model,
        scenario=scenario,
        processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
    )

    assert_frame_equal(
        res_temperature_percentiles.loc[:, exp_temperature_percentiles.columns],
        exp_temperature_percentiles,
        rtol=1e-5,
    )

    metadata_compare_cols = [
        "Category",
        "Category_name",
        "Median peak warming (MAGICCv7.5.3)",
        "Median warming in 2100 (MAGICCv7.5.3)",
        "Exceedance Probability 1.5C (MAGICCv7.5.3)",
        "Exceedance Probability 2.0C (MAGICCv7.5.3)",
    ]
    exp_metadata[list(set(metadata_compare_cols) - {"Category", "Category_name"})] = (
        exp_metadata[
            list(set(metadata_compare_cols) - {"Category", "Category_name"})
        ].astype(float)
    )
    post_processed_metadata.loc[
        exp_metadata["Category"] == "failed-vetting", "Category"
    ] = "failed-vetting"
    post_processed_metadata.loc[
        exp_metadata["Category"] == "failed-vetting", "Category_name"
    ] = "failed-vetting"
    pd.testing.assert_frame_equal(
        post_processed_metadata[metadata_compare_cols],
        exp_metadata[metadata_compare_cols],
        rtol=1e-5,
    )


@pytest.mark.superslow
def test_key_testing_scenarios_all_at_once_parallel(tmp_path):
    infilled_l = []
    exp_temperature_percentiles_l = []
    exp_metadata_l = []
    for model, scenario in KEY_TESTING_MODEL_SCENARIOS:
        mod_scen_metadata = get_ar6_metadata_outputs(
            model=model,
            scenario=scenario,
            ar6_output_data_dir=AR6_OUTPUT_DIR,
        ).loc[[(model, scenario)]]
        if (
            mod_scen_metadata["Median peak warming (MAGICCv7.5.3)"]
            == "no-climate-assessment"
        ).all():
            # Nothing to check against
            continue

        exp_metadata_l.append(mod_scen_metadata)
        infilled_l.append(
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
        exp_temperature_percentiles_l.append(
            get_ar6_temperature_outputs(
                model=model,
                scenario=scenario,
                processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
            )
        )

    infilled = pd.concat(infilled_l)
    exp_temperature_percentiles = pd.concat(exp_temperature_percentiles_l)
    exp_metadata = pd.concat(exp_metadata_l)

    magicc_exe = get_magicc_exe_path()
    scm_runner = AR6SCMRunner.from_ar6_config(
        run_checks=False,
        n_processes=multiprocessing.cpu_count(),
        # progress=False,
        magicc_exe_path=magicc_exe,
        magicc_prob_distribution_path=AR6_MAGICC_PROBABILISTIC_CONFIG_FILE,
        db=OpenSCMDB(
            backend_data=FeatherDataBackend(),
            backend_index=FeatherIndexBackend(),
            db_dir=tmp_path,
        ),
        batch_size_scenarios=5,
    )
    post_processor = AR6PostProcessor.from_ar6_config(
        run_checks=False, n_processes=None
    )

    scm_results = scm_runner(infilled)
    post_processed_timeseries, post_processed_metadata = post_processor(scm_results)

    res_temperature_percentiles = post_processed_timeseries.loc[
        pix.ismatch(
            variable="AR6 climate diagnostics|Surface Temperature (GSAT)|*|*Percentile"
        )
    ]

    assert_frame_equal(
        res_temperature_percentiles.loc[:, exp_temperature_percentiles.columns],
        exp_temperature_percentiles,
        rtol=1e-5,
    )

    metadata_compare_cols = [
        "Category",
        "Category_name",
        "Median peak warming (MAGICCv7.5.3)",
        "Median warming in 2100 (MAGICCv7.5.3)",
        "Exceedance Probability 1.5C (MAGICCv7.5.3)",
        "Exceedance Probability 2.0C (MAGICCv7.5.3)",
    ]
    exp_metadata[list(set(metadata_compare_cols) - {"Category", "Category_name"})] = (
        exp_metadata[
            list(set(metadata_compare_cols) - {"Category", "Category_name"})
        ].astype(float)
    )
    post_processed_metadata.loc[
        exp_metadata["Category"] == "failed-vetting", "Category"
    ] = "failed-vetting"
    post_processed_metadata.loc[
        exp_metadata["Category"] == "failed-vetting", "Category_name"
    ] = "failed-vetting"
    pd.testing.assert_frame_equal(
        post_processed_metadata[metadata_compare_cols],
        exp_metadata[metadata_compare_cols],
        check_like=True,
        rtol=1e-5,
    )
