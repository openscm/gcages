"""
Tests of `gcages.assertions.assert_data_is_all_numeric`
"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from gcages.assertions import DataIsNotAllNumericError, assert_data_is_all_numeric


@pytest.mark.parametrize(
    "inp, exp",
    (
        pytest.param(
            pd.DataFrame(np.arange(12).reshape(4, 3)),
            does_not_raise(),
            id="all-numeric",
        ),
        pytest.param(
            pd.DataFrame(
                [[1, "a"], [2, "b"], [3, "c"], [4, "d"]], columns=["int", "string"]
            ),
            pytest.raises(
                DataIsNotAllNumericError,
                match=re.escape(
                    f"The following columns contain non-numeric data: {('string',)}"
                ),
            ),
            id="string-col",
        ),
        pytest.param(
            pd.DataFrame(
                np.arange(4), columns=["string"], index=pd.Index(["a", "b", "c", "d"])
            ),
            does_not_raise(),
            id="string-index",
        ),
        pytest.param(
            pd.DataFrame(
                np.arange(12).reshape(4, 3),
                columns=[2010.0, 2020.0, 2025.0],
                index=pd.MultiIndex.from_tuples(
                    [
                        (1, "a", 2.0),
                        (2, "b", 1.0),
                        (3, "c", 0.0),
                        (4, "d", -2.0),
                    ],
                    names=["int", "string", "float"],
                ),
            ),
            does_not_raise(),
            id="string-in-multi-index",
        ),
        pytest.param(
            pd.DataFrame(
                np.arange(12).reshape(4, 3),
                columns=[2010.0, 2020.0, 2025.0],
                index=pd.MultiIndex.from_tuples(
                    [
                        (1, "a", 2.0),
                        (2, "b", 1.0),
                        (3, "c", 0.0),
                        (4, "d", -2.0),
                    ],
                    names=["int", "string", "float"],
                ),
            ),
            does_not_raise(),
            id="string-in-multi-index",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    (1, "v1", 1.0, "s1"),
                    (2, "v2", 2.0, "s2"),
                    (3, "v3", 3.0, "s3"),
                    (4, "v4", 4.0, "s4"),
                ],
                columns=[2010.0, "variable", 2025.0, "scenario"],
                index=pd.MultiIndex.from_tuples(
                    [
                        (1, "a", 2.0),
                        (2, "b", 1.0),
                        (3, "c", 0.0),
                        (4, "d", -2.0),
                    ],
                    names=["int", "string", "float"],
                ),
            ),
            pytest.raises(
                DataIsNotAllNumericError,
                match=re.escape(
                    "The following columns contain non-numeric data: "
                    f"{('variable', 'scenario')}"
                ),
            ),
            id="string-in-multi-index",
        ),
    ),
)
def test_assert_data_is_all_numeric(inp, exp):
    with exp:
        assert_data_is_all_numeric(inp)
