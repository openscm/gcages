"""
Test SCM running and post-processing compared to AR6

We can't test SCM running alone
because the output was not saved with AR6.

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
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.ar6 import AR6PostProcessor, AR6SCMRunner
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import (
    assert_frame_equal,
    get_ar6_infilled_emissions,
    get_ar6_metadata_outputs,
    get_ar6_temperature_outputs,
    get_key_testing_model_scenario_parameters,
    get_magicc_exe_path,
)

pix = pytest.importorskip("pandas_indexing")
# Only works if openscm-runner installed
pytest.importorskip("openscm_runner")

AR6_MAGICC_PROBABILISTIC_CONFIG_FILE = (
    Path(__file__).parents[0]
    / "ar6-workflow-inputs"
    / "magicc-ar6-0fd0f62-f023edb-drawnset"
    / "0fd0f62-derived-metrics-id-f023edb-drawnset.json"
)
AR6_OUTPUT_DIR = Path(__file__).parents[0] / "ar6-output"
PROCESSED_AR6_DB_DIR = Path(__file__).parents[0] / "ar6-output-processed"


def strip_off_ar6_infilled_prefix_and_convert_to_gcages(
    indf: pd.DataFrame,
) -> pd.DataFrame:
    indf = update_index_levels_func(
        indf,
        {
            "variable": lambda x: convert_variable_name(
                x.replace("AR6 climate diagnostics|Infilled|", ""),
                from_convention=SupportedNamingConventions.IAMC,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
        copy=False,
    )

    return indf


def get_post_processed_metadata_comparable(indf: pd.DataFrame):
    res = indf.copy()

    # AR6 custom overwrite
    res.loc[res["Category"] == "failed-vetting", "Category"] = "failed-vetting"

    return res


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
    # TODO: renaming
    breakpoint()

    magicc_exe = get_magicc_exe_path()
    scm_runner = AR6SCMRunner.from_ar6_config(
        # Has to be parallel otherwise this is too slow
        n_processes=multiprocessing.cpu_count(),
        progress=False,
        magicc_exe_path=magicc_exe,
        magicc_prob_distribution_path=AR6_MAGICC_PROBABILISTIC_CONFIG_FILE,
    )
    post_processor = AR6PostProcessor.from_ar6_config(n_processes=None)

    scm_results = scm_runner(infilled)
    post_processed = post_processor(scm_results)

    res_temperature_percentiles = post_processed.timeseries.loc[
        pix.ismatch(
            variable="AR6 climate diagnostics|Surface Temperature (GSAT)|*|*Percentile"
        )
    ]

    exp_temperature_percentiles = get_ar6_temperature_outputs(
        model=model,
        scenario=scenario,
        processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
    )

    # TODO: renaming
    breakpoint()
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
    exp_numerical_cols = list(
        set(metadata_compare_cols) - {"Category", "Category_name"}
    )
    exp_metadata[exp_numerical_cols] = exp_metadata[exp_numerical_cols].astype(float)

    post_processed_metadata_comparable = get_post_processed_metadata_comparable(
        post_processed.metadata
    )
    pd.testing.assert_frame_equal(
        post_processed_metadata_comparable[metadata_compare_cols],
        exp_metadata[metadata_compare_cols],
        rtol=1e-5,
    )
