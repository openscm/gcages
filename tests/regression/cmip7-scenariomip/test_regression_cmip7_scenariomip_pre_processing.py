"""
Regression tests of our pre-processing for CMIP7 ScenarioMIP
"""

from pathlib import Path

import pytest
from pandas_openscm.io import load_timeseries_csv

HERE = Path(__file__).parents[0]


@pytest.mark.parametrize(
    "input_file",
    (
        pytest.param(
            HERE / "test-data" / "salted-202504-scenariomip-input.csv",
            id="salted-202504-scenariomip-input",
        ),
    ),
)
def test_pre_processing_regression(input_file, dataframe_regression):
    input_df = load_timeseries_csv(
        input_file,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_column_type=int,
    )

    pre_processor = CMIP7ScenarioMIPPreProcessor.from_ar6_config(
        n_processes=None,  # run serially
    )
    res = pre_processor(input_df)

    dataframe_regression.check(res)
