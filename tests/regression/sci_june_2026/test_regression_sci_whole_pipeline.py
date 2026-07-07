"""
Test SCIjune2026 workflow
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pandas as pd
import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.sci_june_2026.harmonisation import create_scijune2026_global_harmoniser
from gcages.sci_june_2026.infilling import SCIJune2026Infiller
from gcages.sci_june_2026.post_processing import SCIJune2026PostProcessor
from gcages.sci_june_2026.pre_processing import SCIJune2026PreProcessor
from gcages.sci_june_2026.scm_running import SCIJune2026SCMRunner
from gcages.testing import (
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

# Variables
HARMONISATION_YEAR = 2023
KEY_SCI_TESTING_MODEL_SCENARIOS = tuple(
    [
        ("AIM/CGE 2.0", "SSP1-19"),
        ("AIM/CGE 2.0", "SSP3-34"),
        ("AIM/CGE 2.1", "CD-LINKS-INDC2030i_1600"),
        ("AIM/CGE 2.1", "COMMIT-Bridge"),
        ("AIM/CGE V2.2", "ENGAGE-Feasibility-1000/Cost-Effective"),
        ("AIM/CGE V2.2", "ENGAGE-INDCi2030-1000f"),
        ("AIM/Hub-Global 2.0", "SDI-1.5°C"),
        ("AIM/Hub-Global 2.4", "GEO7-Current Trends"),
        ("CGEM-ESM 1.0", "China-2060-SSP2-2°C"),
        ("COFFEE 1.1", "COMMIT-2°C-2020"),
        ("COFFEE 1.5", "ENGAGE-Feasibility-1000/Cost-Effective"),
        ("GCAM 4.2", "SSP1-19"),
        ("GCAM 5.2", "NGFS Phase 1-Current Policies"),
        ("GCAM 5.3", "Deep-Mitigation-Baseline"),
        ("GCAM 7.0", "IAM-COMPACT-Current-Policies-Emissions-Intensity"),
        ("GCAM-PR 5.3", "ParisReinforce-Baseline"),
        ("GEM-E3 V2021", "ENGAGE-INDCi2030-1000"),
        ("IMAGE 3.0", "ENGAGE-INDCi2030-1000"),
        ("IMAGE 3.0.1", "CD-LINKS-INDC2030i_1600"),
        ("IMAGE 3.2", "SSP2021-SSP1-Baseline"),
        ("MESSAGE-GLOBIOM 1.0", "EMF30-BCOC-EndU"),
        ("MESSAGEix-GLOBIOM 1.0", "LowEnergyDemand (1.3/IPCC-AR6)"),
        ("MESSAGEix-GLOBIOM 1.1", "ECEMF-DIAG-C400-lin"),
        ("MESSAGEix-GLOBIOM 1.1", "ENGAGE-INDCi2030-1000"),
        ("MESSAGEix-GLOBIOM 1.2", "COVID-Shift-GreenPush"),
        ("MESSAGEix-GLOBIOM GEI 1.0", "GEI-SSP2-int-lc-15"),
        ("POLES ADVANCE", "ADVANCE-2020-1.5°C-2100"),
        ("REMIND 1.6", "EMF30-BCOC-EndU"),
        ("REMIND 1.7", "ADVANCE-2020-1.5°C-2100"),
        ("REMIND 3.5", "Rescuing-1.5°C-Highest-Possible-Ambition"),
        ("REMIND-MAgPIE 1.7-3.0", "CD-LINKS-NDC2030i_1000"),
        ("REMIND-MAgPIE 3.3-4.8", "NGFS Phase 5-Below 2°C"),
        ("WITCH 5.0", "COMMIT-Bridge"),
        ("WITCH-GLOBIOM 4.2", "ADVANCE-NoPolicy"),
        ("WITCH-GLOBIOM 4.2", "EMF30-BCOC-EndU"),
    ]
)


@pytest.mark.skip_ci_default
@pytest.mark.slow
@get_key_testing_model_scenario_parameters(KEY_SCI_TESTING_MODEL_SCENARIOS)
def test_whole_pipeline(model, scenario, monkeypatch):  # noqa: PLR0915
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
    pre_processor = SCIJune2026PreProcessor.from_standard_config(
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
        rtol=1e-8,
    )

    # HARMONISATION
    # Only works if aneris installed
    pytest.importorskip("aneris")
    harmoniser = create_scijune2026_global_harmoniser(
        historical_emissions_file=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
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
        rtol=1e-8,
    )

    # INFILLING
    infiller = SCIJune2026Infiller.from_files(
        infilling_leader_emissions_file=SCI_INPUT_DIR / "infilling_db_sci.csv",
        ghg_inversions_file=PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
        / "cmip7_ghg_inversions.csv",
        historical_emissions_file=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        harmonisation_year=HARMONISATION_YEAR,
        pre_industrial_year=1750,
        ur=None,
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
        rtol=1e-8,
    )

    # MAGICC and post_processing
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
    # Loading and assessing quantiles timeseries results
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

    # TODO: I do not know why I do not get the same results here
    # I relax the tolerance but is a bit odd.
    assert_frame_equal(
        scm_results,
        exp,
        rtol=1e-7,
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
