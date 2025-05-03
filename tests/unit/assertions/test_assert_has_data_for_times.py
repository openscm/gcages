"""
Tests of `gcages.assertions.assert_has_data_for_times`
"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from gcages.assertions import MissingDataForTimesError, assert_has_data_for_times


def get_df_with_times(times):
    return pd.DataFrame(
        np.zeros((4, len(times))),
        columns=times,
        index=pd.MultiIndex.from_tuples(
            [
                ("sa", "va", "ua"),
                ("sb", "vb", "ub"),
                ("sc", "vc", "uc"),
                ("sd", "vd", "ud"),
            ],
            names=["scenario", "variable", "unit"],
        ),
    )


@pytest.mark.parametrize(
    "inp, expected_times, exp",
    (
        pytest.param(
            get_df_with_times([2020, 2030, 2040]),
            [2020, 2030, 2040],
            does_not_raise(),
            id="all-included",
        ),
        pytest.param(
            get_df_with_times([2020, 2030, 2040]),
            [2020, 2040],
            does_not_raise(),
            id="all-included-subset",
        ),
        pytest.param(
            get_df_with_times([2020, 2025, 2030]),
            [2025],
            does_not_raise(),
            id="all-included-single",
        ),
        pytest.param(
            get_df_with_times([2021, 2022, 2023]),
            [2025, 2030],
            pytest.raises(
                MissingDataForTimesError,
                match=re.escape(
                    "inp is missing data for the following times: "
                    f"{[2025, 2030]}. "
                    "Available times:"
                ),
            ),
            id="all-missing",
        ),
        pytest.param(
            get_df_with_times([2020, 2030]),
            [2020, 2025, 2030],
            pytest.raises(
                MissingDataForTimesError,
                match=re.escape(
                    "inp is missing data for the following times: "
                    f"{[2025]}. "
                    "Available times:"
                ),
            ),
            id="partial-missing-single",
        ),
        pytest.param(
            get_df_with_times([2020, 2030]),
            [2020, 2025, 2028, 2030],
            pytest.raises(
                MissingDataForTimesError,
                match=re.escape(
                    "inp is missing data for the following times: "
                    f"{[2025, 2028]}. "
                    "Available times:"
                ),
            ),
            id="partial-missing-multiple",
        ),
        pytest.param(
            get_df_with_times([2020.0, 2022.5, 2030.0]),
            [2020.0, 2022.5, 2025.0, 2027.5, 2028.0, 2030.0],
            pytest.raises(
                MissingDataForTimesError,
                match=re.escape(
                    "inp is missing data for the following times: "
                    f"{[2025., 2027.5, 2028.]}. "
                    "Available times:"
                ),
            ),
            id="partial-missing-multiple-float",
        ),
    ),
)
def test_assert_has_data_for_times(inp, expected_times, exp):
    with exp:
        assert_has_data_for_times(inp, name="inp", times=expected_times, allow_nan=True)


@pytest.mark.parametrize(
    "inp, expected_times, allow_nan, exp",
    (
        pytest.param(
            pd.DataFrame(
                [
                    [1, 2, 1],
                    [3, 4, 3],
                    [5, 6, 5],
                ],
                columns=[2010, 2020, 2025],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "ua"),
                        ("sb", "vb", "ub"),
                        ("sc", "vc", "uc"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            [2010, 2020, 2025],
            False,
            does_not_raise(),
            id="no-nans",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1, 2, 1],
                    [3, np.nan, 3],
                    [5, 6, 5],
                ],
                columns=[2010, 2020, 2025],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "ua"),
                        ("sb", "vb", "ub"),
                        ("sc", "vc", "uc"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            [2010, 2020, 2025],
            False,
            pytest.raises(
                MissingDataForTimesError,
                match=" ".join(
                    [
                        re.escape(
                            "inp has NaNs for the following times: " f"{[2020]}."
                        ),
                        r".*sb\s*vb\s*ub",
                    ]
                ),
            ),
            id="partial-nans",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1, 2, np.nan],
                    [3, 4, np.nan],
                    [5, 6, np.nan],
                ],
                columns=[2010, 2020, 2025],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "ua"),
                        ("sb", "vb", "ub"),
                        ("sc", "vc", "uc"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            [2010, 2020, 2025],
            False,
            pytest.raises(
                MissingDataForTimesError,
                match=" ".join(
                    [
                        re.escape(
                            "inp has NaNs for the following times: " f"{[2025]}."
                        ),
                        r".*sa\s*va\s*ua",
                        r".*sb\s*vb\s*ub",
                        r".*sc\s*vc\s*uc",
                    ]
                ),
            ),
            id="all-nans",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1, 2, np.nan],
                    [3, np.nan, np.nan],
                    [5, 6, np.nan],
                ],
                columns=[2010, 2020, 2025],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "ua"),
                        ("sb", "vb", "ub"),
                        ("sc", "vc", "uc"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            [2010, 2020, 2025],
            False,
            pytest.raises(
                MissingDataForTimesError,
                match=" ".join(
                    [
                        re.escape(
                            "inp has NaNs for the following times: " f"{[2020, 2025]}."
                        ),
                        r".*sa\s*va\s*ua",
                        r".*sb\s*vb\s*ub",
                        r".*sc\s*vc\s*uc",
                    ]
                ),
            ),
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1, 2, np.nan],
                    [3, np.nan, np.nan],
                    [5, 6, np.nan],
                ],
                columns=[2010, 2020, 2025],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "ua"),
                        ("sb", "vb", "ub"),
                        ("sc", "vc", "uc"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            [2010, 2020, 2025],
            True,
            does_not_raise(),
            id="partial-and-all-nans-allow-nans",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1, 2],
                    [3, np.nan],
                    [5, 6],
                ],
                columns=[2010, 2020],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "ua"),
                        ("sb", "vb", "ub"),
                        ("sc", "vc", "uc"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            [2010, 2020, 2025],
            True,
            pytest.raises(
                MissingDataForTimesError,
                match=re.escape(
                    "inp is missing data for the following times: "
                    f"{[2025]}. "
                    "Available times:"
                ),
            ),
            id="missing-allow-nans",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1, 2],
                    [3, np.nan],
                    [5, 6],
                ],
                columns=[2010, 2020],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "ua"),
                        ("sb", "vb", "ub"),
                        ("sc", "vc", "uc"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            [2010, 2020, 2025],
            False,
            pytest.raises(
                MissingDataForTimesError,
                match=re.escape(
                    "inp is missing data for the following times: "
                    f"{[2025]}. "
                    "Available times:"
                ),
            ),
            id="missing-dont-allow-nans",
        ),
    ),
)
def test_assert_has_data_for_times_nan_handling(inp, expected_times, allow_nan, exp):
    with exp:
        assert_has_data_for_times(
            inp, name="inp", times=expected_times, allow_nan=allow_nan
        )
