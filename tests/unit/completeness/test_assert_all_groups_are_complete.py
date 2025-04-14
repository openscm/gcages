"""
Tests of `gcages.infilling.assert_all_groups_are_complete`
"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from gcages.completeness import NotCompleteError, assert_all_groups_are_complete


@pytest.mark.parametrize(
    "df, complete_index,  exp",
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
            pytest.raises(
                NotCompleteError,
                match="".join(
                    [
                        re.escape(
                            "The DataFrame is not complete. "
                            "The following expected levels are missing:"
                        ),
                        r"\s*.*variable\s*scenario",
                        r"\s*.*va\s*sb\s*",
                        re.escape("The complete index expected for each level is:"),
                    ]
                ),
            ),
            id="missing-timeseries",
        ),
        pytest.param(
            pd.DataFrame(
                np.arange(16).reshape((8, 2)),
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "r1", "W"),
                        ("sa", "vb", "r1", "W"),
                        ("sb", "va", "r1", "W"),
                        ("sb", "vb", "r1", "W"),
                        ("sa", "va", "r2", "W"),
                        ("sa", "vb", "r2", "W"),
                        ("sb", "va", "r2", "W"),
                        ("sb", "vb", "r2", "W"),
                    ],
                    names=["scenario", "variable", "region", "unit"],
                ),
            ),
            pd.MultiIndex.from_product(
                [["va", "vb"], ["r1", "r2"]],
                names=["variable", "region"],
            ),
            does_not_raise(),
            id="infilled-regional",
        ),
        pytest.param(
            pd.DataFrame(
                np.arange(12).reshape((6, 2)),
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "r1", "W"),
                        # ("sa", "vb", "r1", "W"),
                        ("sb", "va", "r1", "W"),
                        ("sb", "vb", "r1", "W"),
                        ("sa", "va", "r2", "W"),
                        ("sa", "vb", "r2", "W"),
                        # ("sb", "va", "r2", "W"),
                        ("sb", "vb", "r2", "W"),
                    ],
                    names=["scenario", "variable", "region", "unit"],
                ),
            ),
            pd.MultiIndex.from_product(
                [["va", "vb"], ["r1", "r2"]],
                names=["variable", "region"],
            ),
            pytest.raises(
                NotCompleteError,
                match="".join(
                    [
                        re.escape(
                            "The DataFrame is not complete. "
                            "The following expected levels are missing:"
                        ),
                        r"\s*.*variable\s*region\s*scenario",
                        r"\s*.*vb\s*r1\s*sa\s*",
                        r"\s*.*va\s*r2\s*sb\s*",
                        re.escape("The complete index expected for each level is:"),
                    ]
                ),
            ),
            id="missing-regional",
        ),
    ),
)
def test_assert_all_groups_are_complete(df, complete_index, exp):
    with exp:
        assert_all_groups_are_complete(df, complete_index=complete_index)
