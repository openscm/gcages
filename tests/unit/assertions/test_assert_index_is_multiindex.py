"""
Tests of `gcages.assertions.assert_index_is_multiindex`
"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from gcages.assertions import IndexIsNotMultiIndexError, assert_index_is_multiindex


@pytest.mark.parametrize(
    "inp, exp",
    (
        pytest.param(
            pd.DataFrame(np.arange(12).reshape(4, 3)),
            pytest.raises(
                IndexIsNotMultiIndexError,
                match=re.escape(
                    "The index is not a `pd.MultiIndex`, "
                    "instead we have type(df.index)="
                ),
            ),
            id="standard-index",
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
            id="multi-index",
        ),
    ),
)
def test_assert_index_is_multiindex(inp, exp):
    with exp:
        assert_index_is_multiindex(inp)
