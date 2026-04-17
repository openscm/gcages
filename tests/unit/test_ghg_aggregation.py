"""
Tests of the `gcages.ghg_aggregation` using ar6 data.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pandas_openscm.io
import pandas_openscm.testing
import pytest

from gcages.ghg_aggregation import calculate_kyoto_ghg

CMIP7_SCENARIOMIP_OUT_DIR = (
    Path(__file__).parents[1]
    / "regression/cmip7-scenariomip"
    / "cmip7-scenariomip-output"
)

pytest.importorskip("openscm_units")

# Tests:
# - works with basic
# - fails if only one of the groups is missing a required timeseries
#   (will require updating climate-processing too)
# - supports different naming conventions
# - works for different gwps
# - out_variable, out_unit, variable_level, unit_level all supported
# - ur can be injected (add ur that knows about GWPZN, then make sure that works)


@pytest.fixture(scope="module")
def indf_basic():
    res = pd.DataFrame(
        [
            [100, 110, 120],
            [10, 11, 12],
            [1000.0, 2000.0, 3000.0],
            [200, 100, 300],
            [5, 6, 7],
            [1000.0, 500.0, 0.0],
        ],
        columns=[2010, 2020, 2025],
        index=pd.MultiIndex.from_tuples(
            [
                ("a", "CO2", "MtCO2 / yr"),
                ("a", "CH4", "MtCH4 / yr"),
                ("a", "N2O", "ktN2O / yr"),
                ("b", "CO2", "MtCO2 / yr"),
                ("b", "CH4", "MtCH4 / yr"),
                ("b", "N2O", "ktN2O / yr"),
            ],
            names=["ms", "variable", "unit"],
        ),
    )
    return res


def test_calculate_kyoto_ghg_basic(indf_basic):
    res = calculate_kyoto_ghg(
        indf_basic,
        kyoto_ghgs=("CO2", "CH4", "N2O"),
    )

    exp = pd.DataFrame(
        [
            [
                100 + 10 * 27.9 + 273 * 1.0,
                110 + 11 * 27.9 + 273 * 2.0,
                120 + 12 * 27.9 + 273 * 3.0,
            ],
            [
                200 + 5 * 27.9 + 273 * 1.0,
                100 + 6 * 27.9 + 273 * 0.5,
                300 + 7 * 27.9 + 273 * 0.0,
            ],
        ],
        columns=[2010, 2020, 2025],
        index=pd.MultiIndex.from_tuples(
            [
                ("a", "Kyoto GHG", "MtCO2 / yr"),
                ("b", "Kyoto GHG", "MtCO2 / yr"),
            ],
            names=["ms", "variable", "unit"],
        ),
    )

    pandas_openscm.testing.assert_frame_alike(res, exp)


def test_calculate_kyoto_ghg_all_missing(indf_basic):
    indf = indf_basic.loc[~(indf_basic.index.get_level_values("variable") == "N2O")]
    error_msg = re.escape(
        "You are missing the following Kyoto GHGs: {'N2O'}. "
        "Please either supply these gases "
        "or provide a different value for `kyoto_ghgs` to `calculate_kyoto_ghg`. "
        "Currently kyoto_ghgs=('CO2', 'CH4', 'N2O')."
    )
    with pytest.raises(AssertionError, match=error_msg):
        calculate_kyoto_ghg(
            indf,
            kyoto_ghgs=("CO2", "CH4", "N2O"),
        )


def test_calculate_kyoto_ghg_one_missing_error(indf_basic):
    indf = indf_basic.loc[
        ~(
            (indf_basic.index.get_level_values("variable") == "N2O")
            & (indf_basic.index.get_level_values("ms") == "b")
        )
    ]
    error_msg = re.escape(
        "For some groups, you are missing some Kyoto GHGs. "
        "Please either supply these gases for these groups "
        "or provide a different value for `kyoto_ghgs` to `calculate_kyoto_ghg`. "
        "Currently kyoto_ghgs=('CO2', 'CH4', 'N2O'). "
        "The groups and their missing Kyoto GHGs are: "
    )
    with pytest.raises(AssertionError, match=error_msg):
        calculate_kyoto_ghg(
            indf,
            kyoto_ghgs=("CO2", "CH4", "N2O"),
        )
