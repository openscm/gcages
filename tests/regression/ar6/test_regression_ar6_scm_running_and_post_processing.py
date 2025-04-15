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

import numpy as np
import pandas as pd
import pytest
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.ar6 import AR6PostProcessor, AR6SCMRunner, get_ar6_full_historical_emissions
from gcages.ar6.post_processing import set_new_single_value_levels
from gcages.post_processing import PostProcessingResult
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import (
    KEY_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_ar6_infilled_emissions,
    get_ar6_metadata_outputs,
    get_ar6_temperature_outputs,
    get_key_testing_model_scenario_parameters,
    guess_magicc_exe_path,
)
from gcages.units_helpers import strip_pint_incompatible_characters_from_unit_string

pix = pytest.importorskip("pandas_indexing")
# Only works if openscm-runner installed
pytest.importorskip("openscm_runner.adapters")

AR6_INFILLING_DB_CFCS_FILE = (
    Path(__file__).parents[0] / "ar6-workflow-inputs" / "infilling_db_ar6_cfcs.csv"
)
AR6_MAGICC_PROBABILISTIC_CONFIG_FILE = (
    Path(__file__).parents[0]
    / "ar6-workflow-inputs"
    / "magicc-ar6-0fd0f62-f023edb-drawnset"
    / "0fd0f62-derived-metrics-id-f023edb-drawnset.json"
)
AR6_OUTPUT_DIR = Path(__file__).parents[0] / "ar6-output"
PROCESSED_AR6_DB_DIR = Path(__file__).parents[0] / "ar6-output-processed"


def strip_off_ar6_infilled_prefix_and_convert_to_gcages_and_fix_units(
    indf: pd.DataFrame,
) -> pd.DataFrame:
    indf = update_index_levels_func(
        indf,
        {
            "variable": lambda x: convert_variable_name(
                x.replace("AR6 climate diagnostics|Infilled|", ""),
                from_convention=SupportedNamingConventions.IAMC,
                to_convention=SupportedNamingConventions.GCAGES,
            ),
            "unit": lambda x: strip_pint_incompatible_characters_from_unit_string(
                x
            ).replace("HFC245ca", "HFC245fa"),
        },
        copy=False,
    )

    return indf


def convert_to_ar6_percentile_output(indf: pd.DataFrame) -> pd.DataFrame:
    res = indf.reset_index("quantile")
    res["percentile"] = (res["quantile"] * 100.0).round(1).astype(str)
    res = res.set_index("percentile", append=True).drop("quantile", axis="columns")

    res = res.pix.format(
        variable="AR6 climate diagnostics|{variable}|{climate_model}|{percentile}th Percentile",  # noqa: E501
        drop=True,
    )

    res.columns = res.columns.astype(indf.columns.dtype)

    return res


def get_post_processed_metadata_comparable(res_pp: PostProcessingResult):
    out_index = ["model", "scenario"]

    # Works only for MAGICC, others need the category in their name
    categories = res_pp.metadata_categories.unstack("metric")
    categories.columns = categories.columns.str.capitalize()
    categories = categories.reset_index(
        categories.index.names.difference(out_index), drop=True
    )

    exceedance_probs_s = update_index_levels_func(
        res_pp.metadata_exceedance_probabilities,
        {"threshold": lambda x: np.round(x, 1)},
    )
    exceedance_probs_s = exceedance_probs_s.pix.format(
        out_name="Exceedance Probability {threshold}C ({climate_model})"
    )
    exceedance_probs = exceedance_probs_s.reset_index(
        exceedance_probs_s.index.names.difference([*out_index, "out_name"]), drop=True
    ).unstack("out_name")
    exceedance_probs = exceedance_probs / 100.0

    def get_out_quantile(q: float) -> str:
        if q == 0.5:
            return "Median"

        return f"P{q*100:.0f}"

    quantile_metadata_l = []
    for v_str, metric_id in (
        ("peak warming", "max"),
        ("warming in 2100", 2100),
        ("year of peak warming", "max_year"),
    ):
        start = res_pp.metadata_quantile[
            res_pp.metadata_quantile.index.get_level_values("metric") == metric_id
        ]
        tmp_a = update_index_levels_func(start, {"quantile": get_out_quantile})
        tmp_b = set_new_single_value_levels(tmp_a, {"v_str": v_str}, copy=False)
        tmp_c = tmp_b.pix.format(out_name="{quantile} {v_str} ({climate_model})")
        tmp_d = tmp_c.reset_index(
            tmp_a.index.names.difference([*out_index, "out_name"]), drop=True
        ).unstack("out_name")

        quantile_metadata_l.append(tmp_d)

    quantile_metadata = pd.concat(quantile_metadata_l, axis="columns")

    out = pd.concat([categories, exceedance_probs, quantile_metadata], axis="columns")

    return out


@pytest.mark.skip_ci_default
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

    infilled = strip_off_ar6_infilled_prefix_and_convert_to_gcages_and_fix_units(
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

    magicc_exe = guess_magicc_exe_path()
    scm_runner = AR6SCMRunner.from_ar6_config(
        # Has to be parallel otherwise this is too slow
        n_processes=multiprocessing.cpu_count(),
        progress=False,
        magicc_exe_path=magicc_exe,
        magicc_prob_distribution_path=AR6_MAGICC_PROBABILISTIC_CONFIG_FILE,
        historical_emissions=get_ar6_full_historical_emissions(
            AR6_INFILLING_DB_CFCS_FILE
        ),
        harmonisation_year=2015,
        output_variables=("Surface Air Temperature Change",),
    )
    post_processor = AR6PostProcessor.from_ar6_config(n_processes=None)

    scm_results = scm_runner(infilled)
    post_processed = post_processor(scm_results)

    res_temperature_percentiles_comparable = convert_to_ar6_percentile_output(
        post_processed.timeseries_quantile.loc[
            pix.ismatch(variable="Surface Temperature (GSAT)")
        ]
    )

    exp_temperature_percentiles = get_ar6_temperature_outputs(
        model=model,
        scenario=scenario,
        processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
    )

    assert_frame_equal(
        res_temperature_percentiles_comparable.loc[
            :, exp_temperature_percentiles.columns
        ],
        exp_temperature_percentiles,
        rtol=1e-5,
    )

    metadata_compare_cols = [
        "Category",
        "Category_name",
        "Median peak warming (MAGICCv7.5.3)",
        "P33 peak warming (MAGICCv7.5.3)",
        "Median warming in 2100 (MAGICCv7.5.3)",
        "Median year of peak warming (MAGICCv7.5.3)",
        "Exceedance Probability 1.5C (MAGICCv7.5.3)",
        "Exceedance Probability 2.0C (MAGICCv7.5.3)",
    ]
    exp_numerical_cols = list(
        set(metadata_compare_cols) - {"Category", "Category_name"}
    )
    exp_metadata[exp_numerical_cols] = exp_metadata[exp_numerical_cols].astype(float)

    post_processed_metadata_comparable = get_post_processed_metadata_comparable(
        post_processed
    )
    # If needed, use the failed vetting flag
    post_processed_metadata_comparable.loc[
        exp_metadata["Category"] == "failed-vetting",
        ["Category", "Category_name"],
    ] = "failed-vetting"
    pd.testing.assert_frame_equal(
        post_processed_metadata_comparable[metadata_compare_cols],
        exp_metadata[metadata_compare_cols],
        rtol=1e-5,
    )


@pytest.mark.slow
def test_parallel(tmp_path):
    """Test a few scenarios in parallel, not all to save compute time"""
    # Required for progress bars
    pytest.importorskip("tqdm.auto")
    # Required for database
    pytest.importorskip("filelock")

    infilled_l = []
    exp_temperature_percentiles_l = []
    exp_metadata_l = []
    for model, scenario in KEY_TESTING_MODEL_SCENARIOS[:3]:
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
        exp_metadata_l.append(
            get_ar6_metadata_outputs(
                model=model,
                scenario=scenario,
                ar6_output_data_dir=AR6_OUTPUT_DIR,
            )
        )

    infilled = strip_off_ar6_infilled_prefix_and_convert_to_gcages_and_fix_units(
        pd.concat(infilled_l)
    )
    exp_temperature_percentiles = pd.concat(exp_temperature_percentiles_l)
    exp_metadata = pd.concat(exp_metadata_l)

    magicc_exe = guess_magicc_exe_path()
    scm_runner = AR6SCMRunner.from_ar6_config(
        n_processes=multiprocessing.cpu_count(),
        # run with progress bars is the default
        # progress=False,
        # Use a db for these runs
        db=OpenSCMDB(
            backend_data=FeatherDataBackend(),
            backend_index=FeatherIndexBackend(),
            db_dir=tmp_path,
        ),
        # Force some batching too
        batch_size_scenarios=2,
        magicc_exe_path=magicc_exe,
        magicc_prob_distribution_path=AR6_MAGICC_PROBABILISTIC_CONFIG_FILE,
        historical_emissions=get_ar6_full_historical_emissions(
            AR6_INFILLING_DB_CFCS_FILE
        ),
        harmonisation_year=2015,
        output_variables=("Surface Air Temperature Change",),
    )
    post_processor = AR6PostProcessor.from_ar6_config(n_processes=None)

    scm_runner(infilled)
    # Make sure the caching works
    # (this should not run anything, but should return the right results)
    scm_results = scm_runner(infilled)
    post_processed = post_processor(scm_results)

    res_temperature_percentiles_comparable = convert_to_ar6_percentile_output(
        post_processed.timeseries_quantile.loc[
            pix.ismatch(variable="Surface Temperature (GSAT)")
        ]
    )
    assert_frame_equal(
        res_temperature_percentiles_comparable.loc[
            :, exp_temperature_percentiles.columns
        ],
        exp_temperature_percentiles,
        rtol=1e-5,
    )

    metadata_compare_cols = [
        "Category",
        "Category_name",
        "Median peak warming (MAGICCv7.5.3)",
        "P33 peak warming (MAGICCv7.5.3)",
        "Median warming in 2100 (MAGICCv7.5.3)",
        "Median year of peak warming (MAGICCv7.5.3)",
        "Exceedance Probability 1.5C (MAGICCv7.5.3)",
        "Exceedance Probability 2.0C (MAGICCv7.5.3)",
    ]
    exp_numerical_cols = list(
        set(metadata_compare_cols) - {"Category", "Category_name"}
    )
    exp_metadata[exp_numerical_cols] = exp_metadata[exp_numerical_cols].astype(float)

    post_processed_metadata_comparable = get_post_processed_metadata_comparable(
        post_processed
    )
    pd.testing.assert_frame_equal(
        post_processed_metadata_comparable[metadata_compare_cols],
        exp_metadata[metadata_compare_cols],
        rtol=1e-5,
    )
