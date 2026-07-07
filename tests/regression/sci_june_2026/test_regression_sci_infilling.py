"""
Test infilling compared to SCI June 2026
(commit: 94025c1247c21c96961f50e9388ee404bb4ae752)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.sci_june_2026.infilling import SCIJune2026Infiller
from gcages.testing import (
    KEY_SCI_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_key_testing_model_scenario_parameters,
)

pix = pytest.importorskip("pandas_indexing")

SCI_INPUT_DIR = Path(__file__).parents[0] / "sci_workflow_inputs"
SCI_OUTPUT_DIR = Path(__file__).parents[0] / "sci_workflow_expected_outputs"

PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR = (
    Path(__file__).parents[1] / "cmip7-scenariomip/cmip7-scenariomip-workflow-inputs"
)
CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "history_cmip7_scenariomip.csv"
)

HARMONISATION_YEAR = 2023


@pytest.mark.parametrize(
    "input_file",
    (
        pytest.param(
            Path(__file__).parents[0]
            / "sci_workflow_expected_outputs"
            / "sci_harmonised.csv",
            id="sci_harmonised",
        ),
    ),
)
@get_key_testing_model_scenario_parameters(KEY_SCI_TESTING_MODEL_SCENARIOS)
def test_individual_scenario_class(input_file, model, scenario):
    # Load harmonised results
    input_df = load_timeseries_csv(
        input_file,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_columns_type=int,
    )
    harmonised = input_df.loc[pix.ismatch(model=model, scenario=scenario)]

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

    # Load complete results
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
    if exp.empty:
        raise AssertionError

    assert_frame_equal(complete, exp)
