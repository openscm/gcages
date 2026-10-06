"""
Tests of the `gcages.interpolation`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gcages.interpolation import interpolate_to_annual


@pytest.fixture(scope="module")
def indf_basic():
    res = pd.DataFrame(
        [
            [100, np.nan, 120],
            [np.nan, 11, 12],
            [1000.0, 2000.0, np.nan],
            [200, np.nan, np.nan],
        ],
        columns=[2010, 2012, 2015],
        index=pd.MultiIndex.from_tuples(
            [
                ("a", "CO2", "MtCO2 / yr"),
                ("a", "CH4", "MtCH4 / yr"),
                ("b", "CO2", "MtCO2 / yr"),
                ("b", "CH4", "MtCH4 / yr"),
            ],
            names=["ms", "variable", "unit"],
        ),
    )

    return res


def test_interpolate_to_annual_basic(indf_basic):
    res = interpolate_to_annual(indf_basic)

    exp = pd.DataFrame(
        [
            # Gap filled linearly between 2010 and 2015
            [100.0, 104.0, 108.0, 112.0, 116.0, 120.0],
            # Leading NaNs are left as NaN
            [np.nan, np.nan, 11.0, 11.0 + 1 / 3, 11.0 + 2 / 3, 12.0],
            # Trailing NaNs are forward-filled with the last value
            [1000.0, 1500.0, 2000.0, 2000.0, 2000.0, 2000.0],
            [200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
        ],
        columns=[2010, 2011, 2012, 2013, 2014, 2015],
        index=pd.MultiIndex.from_tuples(
            [
                ("a", "CO2", "MtCO2 / yr"),
                ("a", "CH4", "MtCH4 / yr"),
                ("b", "CO2", "MtCO2 / yr"),
                ("b", "CH4", "MtCH4 / yr"),
            ],
            names=["ms", "variable", "unit"],
        ),
    )

    pd.testing.assert_frame_equal(res, exp)


def test_interpolate_to_annual_end_year_beyond_data_is_forward_filled(indf_basic):
    res = interpolate_to_annual(indf_basic, end_year=2017)

    assert res.columns.max() == 2017
    # Constant extrapolation from the last value
    assert (res.loc[:, 2016:2017].T == res[2015]).all().all()


def test_interpolate_to_annual_start_year_before_data_stays_nan(indf_basic):
    res = interpolate_to_annual(indf_basic, start_year=2008)

    assert res.loc[:, 2008:2009].isnull().all().all()


def test_interpolate_to_annual_no_columns():
    df = pd.DataFrame(index=pd.Index(["a", "b"], name="variable"))

    with pytest.raises(ValueError, match="has no columns"):
        interpolate_to_annual(df)


def test_interpolate_to_annual_wrong_years(indf_basic):

    with pytest.raises(ValueError, match="must be <="):
        interpolate_to_annual(indf_basic, 2018, 2000)
