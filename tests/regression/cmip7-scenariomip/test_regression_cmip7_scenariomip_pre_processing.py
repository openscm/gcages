"""
Regression tests of our pre-processing for CMIP7 ScenarioMIP
"""

from pathlib import Path

import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor

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
@pytest.mark.xfail(reason="no valid salted data yet")
def test_pre_processing_regression(input_file, dataframe_regression):
    input_df = load_timeseries_csv(
        input_file,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_column_type=int,
    )
    input_df.columns.name = "year"

    pre_processor = CMIP7ScenarioMIPPreProcessor(
        n_processes=None,  # run serially
    )
    # TODO: find some non-broken data to salt
    res = pre_processor(input_df)

    for attr in [
        "global_workflow_emissions",
        "region_sector_workflow_emissions",
        "reaggregated_emissions",
    ]:
        dataframe_regression.check(
            getattr(res, attr), basename=f"{input_file.stem}_attr"
        )
