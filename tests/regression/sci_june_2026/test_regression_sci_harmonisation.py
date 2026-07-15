"""
Test harmonisation compared to notebooks used for SCI June 2026

Note that you could use this to test all scenarios,
but we don't to save computational resources.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.sci_june_2026.harmonisation import create_scijune2026_harmoniser
from gcages.testing import (
    KEY_SCI_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_key_testing_model_scenario_parameters,
)

pix = pytest.importorskip("pandas_indexing")

SCI_INPUT_DIR = Path(__file__).parents[0] / "sci_workflow_inputs"
SCI_OUTPUT_DIR = Path(__file__).parents[0] / "sci_workflow_expected_outputs"

PROCESSED_SCI_INPUT_DIR = (
    Path(__file__).parents[1] / "cmip7-scenariomip/cmip7-scenariomip-workflow-inputs"
)
SCI_HISTORICAL_EMISSIONS_FILE = (
    PROCESSED_SCI_INPUT_DIR / "history_cmip7_scenariomip.csv"
)
HARMONISATION_YEAR = 2023


@get_key_testing_model_scenario_parameters(KEY_SCI_TESTING_MODEL_SCENARIOS)
def test_individual_scenario_global(model, scenario):
    input_file = (
        Path(__file__).parents[0]
        / "sci_workflow_expected_outputs"
        / "sci_pre-processed.csv"
    )
    input_df = load_timeseries_csv(
        input_file,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_columns_type=int,
    )

    pre_processed = input_df.loc[pix.ismatch(model=model, scenario=scenario)]
    # Harmonise
    # Only works if aneris installed
    pytest.importorskip("aneris")
    harmoniser = create_scijune2026_harmoniser(
        historical_emissions_file=SCI_HISTORICAL_EMISSIONS_FILE,
        aneris_overrides_file=SCI_INPUT_DIR / "sci_overrides.csv",
        harmonisation_year=HARMONISATION_YEAR,
        n_processes=None,  # not parallel
        progress=False,
    )
    res = harmoniser(pre_processed)

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
    if exp.empty:
        raise AssertionError

    assert_frame_equal(res, exp)


@pytest.mark.slow
def test_key_testing_scenarios_all_at_once_parallel():
    input_file = (
        Path(__file__).parents[0]
        / "sci_workflow_expected_outputs"
        / "sci_pre-processed.csv"
    )
    pre_processed = load_timeseries_csv(
        input_file,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )

    # Harmonise
    # Only works if aneris installed
    pytest.importorskip("aneris")
    harmoniser = create_scijune2026_harmoniser(
        historical_emissions_file=SCI_HISTORICAL_EMISSIONS_FILE,
        aneris_overrides_file=SCI_INPUT_DIR / "sci_overrides.csv",
        harmonisation_year=HARMONISATION_YEAR,
        n_processes=None,  # not parallel
        progress=False,
    )
    res = harmoniser(pre_processed)

    # Loading and assessing pre_processed results
    file = SCI_OUTPUT_DIR / "sci_harmonised.csv"
    exp = load_timeseries_csv(
        file,
        index_columns=[
            "model",
            "scenario",
            "region",
            "variable",
            "unit",
        ],
        out_columns_type=int,
    )
    if exp.empty:
        raise AssertionError

    assert_frame_equal(res, exp)
