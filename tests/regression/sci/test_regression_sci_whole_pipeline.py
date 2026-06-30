"""
Test infilling compared for CMIP7 ScenarioMIP
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pandas as pd
import pytest

from gcages.sci_june_2026.harmonisation import SCIHarmoniser
from gcages.sci_june_2026.infilling import SCIInfiller
from gcages.sci_june_2026.pre_processing import SCIPreProcessor
from gcages.sci_june_2026.scm_running import SCISCMRunner
from gcages.testing import (
    assert_frame_equal,
    guess_magicc_exe,
)

pix = pytest.importorskip("pandas_indexing")

SCI_INPUT_DIR = Path(__file__).parents[0] / "sci_workflow_inputs"
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

HARMONISATION_YEAR = 2023


@pytest.mark.skip_ci_default
@pytest.mark.slow
def test_whole_pipeline(monkeypatch):
    """Test a few scenarios, not all to save compute time"""
    # LOADING SCENARIO
    file = SCI_INPUT_DIR / "SCI-2025_v1.0_pathways_ensemble_global.xlsx"

    input_df = pd.read_excel(file, sheet_name="data")

    input_df.columns = input_df.columns.str.lower()
    input_df = input_df.set_index(["model", "scenario", "region", "variable", "unit"])
    emissions = input_df.loc[pix.ismatch(variable="Emissions**", region="World")]
    emissions.columns = emissions.columns.astype(int)

    emissions = emissions.sort_index(axis="columns")

    pre_processor = SCIPreProcessor.from_standard_config(
        harmonisation_year=HARMONISATION_YEAR,
        n_processes=None,  # run serially
        progress=False,
        run_checks=True,
    )
    pre_processed = pre_processor(emissions)

    if pre_processed.empty:
        raise AssertionError

    # HARMONISATION
    # Harmonise
    # Only works if aneris installed
    pytest.importorskip("aneris")
    harmoniser = SCIHarmoniser.from_files(
        historical_emissions_file=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        aneris_overrides_file=SCI_INPUT_DIR / "sci_overrides.csv",
        harmonisation_year=HARMONISATION_YEAR,
    )
    harmonised = harmoniser(pre_processed)

    # INFILLING
    infiller = SCIInfiller.from_files(
        infilling_leader_emissions_file=SCI_INPUT_DIR / "infilling_db_sci.csv",
        ghg_inversions_file=PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
        / "cmip7_ghg_inversions.csv",
        historical_emissions_file=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        harmonisation_year=HARMONISATION_YEAR,
        pi_year=1750,
        ur=None,
    )
    complete = infiller(harmonised)

    # MAGICC and post_processing
    monkeypatch.delenv("MAGICC_EXECUTABLE_7", raising=False)
    scm_runner = SCISCMRunner.from_files(
        magicc_exe_path=guess_magicc_exe(CMIP7_SCENARIOMIP_MAGICC_EXECUTABLES_DIR),
        magicc_prob_distribution_path=CMIP7_SCENARIOMIP_MAGICC_PROBABILISTIC_CONFIG_FILE,
        output_variables=("Surface Air Temperature Change",),
        historical_emissions_path=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        harmonisation_year=HARMONISATION_YEAR,
        n_processes=multiprocessing.cpu_count(),
    )

    scm_results = scm_runner(complete)
    assert_frame_equal(scm_results, scm_results)

    # post_processor = CMIP7ScenarioMIPPostProcessor.from_cmip7_scenariomip_config()
    # post_processed = post_processor(scm_results)
