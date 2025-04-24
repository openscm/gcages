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
                [["r1", "r2"], ["va", "vb"]],
                names=["region", "variable"],
            ),
            does_not_raise(),
            id="infilled-regional-differing-order",
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
        pytest.param(
            pd.DataFrame(
                np.arange(2).reshape((1, 2)),
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "r1", "W"),
                    ],
                    names=["scenario", "variable", "region", "unit"],
                ),
            ).iloc[:0, :],
            pd.MultiIndex.from_product(
                [["va", "vb"], ["r1", "r2"]],
                names=["variable", "region"],
            ),
            pytest.raises(ValueError, match=re.escape("`to_check` is empty")),
            id="empty-input",
        ),
    ),
)
def test_assert_all_groups_are_complete(df, complete_index, exp):
    with exp:
        assert_all_groups_are_complete(df, complete_index=complete_index)


def test_assert_all_groups_are_complete_user_unit_col():
    start = pd.DataFrame(
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
            names=["scenario", "variable", "units"],
        ),
    )
    complete_index = pd.MultiIndex.from_tuples(
        [
            ("va",),
            ("vb",),
        ],
        names=["variable"],
    )

    # If you forget the unit col, you get an error
    with pytest.raises(
        KeyError,
        match=re.escape(
            "unit_col='unit' is not in "
            "to_check.index.names=FrozenList(['scenario', 'variable', 'units'])"
        ),
    ):
        assert_all_groups_are_complete(start, complete_index)

    # If you specify the units, all is happy
    assert_all_groups_are_complete(start, complete_index, unit_col="units")


def test_assert_all_groups_are_complete_user_group_keys():
    start = pd.DataFrame(
        [
            [1.0, 2.0],
            [3.0, 2.0],
            [1.0, 2.0],
            [3.0, 2.0],
        ],
        columns=[2015, 2100],
        index=pd.MultiIndex.from_tuples(
            [
                ("sa", "va", "W", "g1"),
                ("sa", "vb", "W", "g2"),
                ("sb", "va", "W", "g1"),
                ("sb", "vb", "W", "g2"),
            ],
            names=["scenario", "variable", "unit", "unhelpful_grouper"],
        ),
    )
    complete_index = pd.MultiIndex.from_tuples(
        [
            ("va",),
            ("vb",),
        ],
        names=["variable"],
    )

    # If you forget the unit col, you get unhelpful groupings hence an error
    with pytest.raises(NotCompleteError):
        assert_all_groups_are_complete(start, complete_index)

    # If you specify the groups, all is happy
    assert_all_groups_are_complete(start, complete_index, group_keys=["scenario"])
