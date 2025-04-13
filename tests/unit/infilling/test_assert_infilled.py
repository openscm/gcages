"""
Tests of `gcages.infilling.assert_infilled`
"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from gcages.infilling import NotInfilledError, assert_infilled


def get_df(index):
    return pd.DataFrame(
        np.zeros((index.shape[0], 3)),
        columns=range(3),
        index=index,
    )


@pytest.mark.parametrize(
    "df, full_emissions_index,  exp",
    (
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.0],
                    [3.0, 2.0],
                    [1.0, 2.0],
                    [3.0, 2.0],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "W"),
                        ("sa", "vb", "W"),
                        ("sb", "va", "W"),
                        ("sb", "vb", "W"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            pd.MultiIndex.from_tuples(
                [
                    ("va",),
                    ("vb",),
                ],
                names=["variable"],
            ),
            does_not_raise(),
            id="infilled",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.0],
                    [3.0, 2.0],
                    # [1.0, 2.0],
                    [3.0, 2.0],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "W"),
                        ("sa", "vb", "W"),
                        # ("sb", "va", "W"),
                        ("sb", "vb", "W"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            pd.MultiIndex.from_tuples(
                [
                    ("va",),
                    ("vb",),
                ],
                names=["variable"],
            ),
            pytest.raises(NotInfilledError, match=re.escape("junk")),
            id="missing-timeseries",
        ),
        # TODO: infilled-regional
    ),
)
def test_assert_infilled(df, full_emissions_index, exp):
    assert_infilled(df, full_emissions_index=full_emissions_index)
    with exp:
        assert_infilled(df, full_emissions_index=full_emissions_index)
