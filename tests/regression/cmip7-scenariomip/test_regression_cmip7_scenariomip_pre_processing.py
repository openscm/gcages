"""
Regression tests of our pre-processing for CMIP7 ScenarioMIP
"""

from pathlib import Path

import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
from gcages.cmip7_scenariomip.pre_processing.reaggregation import ReaggregatorBasic

HERE = Path(__file__).parents[0]

# Need to split the sectors etc.
pytest.importorskip("pandas_indexing")


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
    input_df.columns.name = "year"

    # In case the new salted data needs a bit of make-up
    # mask = input_df.index.get_level_values("variable").str.startswith(
    #     "Emissions"
    # ) | input_df.index.get_level_values("variable").str.startswith("Carbon Removal")
    # input_df = input_df[mask]
    # input_df = input_df.loc[:, 2015:2100:1].dropna(how="all", axis="columns")
    # input_df = input_df.T.interpolate(method="index").T
    #
    # input_df = input_df.rename(
    #     lambda x: re.sub(r"^Carbon Removal", r"Carbon Removal|CO2", x)
    #     if isinstance(x, str)
    #     else x,
    #     level="variable",
    # )
    # input_df.to_csv("salted.csv")

    model_regions = [
        r
        for r in input_df.index.get_level_values("region").unique()
        if r.startswith("model_1")
    ]

    reaggregator = ReaggregatorBasic(model_regions=model_regions)

    pre_processor = CMIP7ScenarioMIPPreProcessor(
        reaggregator=reaggregator,
        n_processes=None,  # run serially
        progress=False,
        # TODO: turn this to True when cleaning up the CDR handling
        run_checks=False,
    )

    res = pre_processor(input_df)

    for attr in [
        "assumed_zero_emissions",
        "global_workflow_emissions",
        "global_workflow_emissions_raw_names",
        "gridding_workflow_emissions",
    ]:
        # Interestingly, this won't fail if there are extra, unexpected columns
        # in the regression data against which we are comparing.
        dataframe_regression.check(
            getattr(res, attr), basename=f"{input_file.stem}_{attr}"
        )
