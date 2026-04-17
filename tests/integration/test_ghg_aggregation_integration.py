"""
Tests of the `gcages.ghg_aggregation` using ar6 data.
"""

from __future__ import annotations

from pathlib import Path

import pandas_openscm.io
import pandas_openscm.testing
import pytest

from gcages.ghg_aggregation import calculate_kyoto_ghg
from gcages.renaming import SupportedNamingConventions
from gcages.testing import (
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS,
    get_key_testing_model_scenario_parameters,
)

CMIP7_SCENARIOMIP_OUT_DIR = (
    Path(__file__).parents[1]
    / "regression/cmip7-scenariomip"
    / "cmip7-scenariomip-output"
)

pytest.importorskip("openscm_units")


@get_key_testing_model_scenario_parameters(
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS
)
@pytest.mark.parametrize("gwp", ("AR6GWP100",))
def test_calculate_kyoto_ghg_scenariomip_like(model, scenario, gwp):
    complete = pandas_openscm.io.load_timeseries_csv(
        CMIP7_SCENARIOMIP_OUT_DIR / f"{model}_{scenario}_complete.csv",
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )

    out_variable = f"Emissions|Kyoto GHG {gwp}"
    res = calculate_kyoto_ghg(
        complete,
        indf_naming_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        gwp=gwp,
        out_variable=out_variable,
    )

    exp = pandas_openscm.io.load_timeseries_csv(
        CMIP7_SCENARIOMIP_OUT_DIR / f"{model}_{scenario}_complete_ghg-aggregates.csv",
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    ).xs(out_variable, level="variable", drop_level=False)

    pandas_openscm.testing.assert_frame_alike(res, exp)
