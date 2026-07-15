"""
Regression tests of our pre-processing for CMIP7 ScenarioMIP
"""

from pathlib import Path

import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.sci_june_2026.pre_processing import SCIJune2026PreProcessor
from gcages.testing import (
    KEY_SCI_TESTING_MODEL_SCENARIOS,
    assert_frame_equal,
    get_key_testing_model_scenario_parameters,
)

HERE = Path(__file__).parents[0]
SCI_OUTPUT_DIR = HERE / "sci_workflow_expected_outputs"

# Need to split the sectors etc.
pix = pytest.importorskip("pandas_indexing")


@get_key_testing_model_scenario_parameters(KEY_SCI_TESTING_MODEL_SCENARIOS)
def test_pre_processing_regression(model, scenario):
    input_file = (
        HERE / "sci_workflow_inputs" / "SCI-2026-June-unique-testing-pathways.csv"
    )

    input_df = load_timeseries_csv(
        input_file,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_columns_type=int,
    )
    emissions = input_df.loc[
        pix.ismatch(
            model=model, scenario=scenario, variable="Emissions**", region="World"
        )
    ]

    pre_processor = SCIJune2026PreProcessor.from_sci_june2026_config(
        n_processes=None,  # run serially
        progress=False,
        run_checks=True,
    )
    res = pre_processor(emissions)

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
        res.loc[:, 2023:],
        exp,
        rtol=1e-7,
    )
