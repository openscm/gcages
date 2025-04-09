"""
Tests of `gcages.assertions.assert_df_has_index_levels`
"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from gcages.assertions import MissingIndexLevelsError, assert_has_index_levels


def get_df_with_index_levels(index_levels):
    return pd.DataFrame(
        np.arange(12).reshape(4, 3),
        columns=[2010.0, 2020.0, 2025.0],
        index=pd.MultiIndex.from_tuples(
            [tuple(v for v in range(len(index_levels))) for _ in range(4)],
            names=index_levels,
        ),
    )


@pytest.mark.parametrize(
    "inp, expected_levels, exp",
    (
        pytest.param(
            get_df_with_index_levels(["int", "string", "float"]),
            ["int", "string", "float"],
            does_not_raise(),
            id="all-included",
        ),
        pytest.param(
            get_df_with_index_levels(["int", "string", "float"]),
            ["string", "float"],
            does_not_raise(),
            id="all-included-subset",
        ),
        pytest.param(
            get_df_with_index_levels(["int", "string", "float"]),
            ["string"],
            does_not_raise(),
            id="all-included-single",
        ),
        pytest.param(
            get_df_with_index_levels(["int", "string", "float"]),
            ["all", "missing"],
            pytest.raises(
                MissingIndexLevelsError,
                match=re.escape(
                    "The DataFrame is missing the following index levels: "
                    f"{['all', 'missing']}. "
                    "Available index levels: "
                    f"{['int', 'string', 'float']}"
                ),
            ),
            id="all-missing",
        ),
        pytest.param(
            get_df_with_index_levels(["int", "string", "float"]),
            ["missing", "int", "float"],
            pytest.raises(
                MissingIndexLevelsError,
                match=re.escape(
                    "The DataFrame is missing the following index levels: "
                    f"{['missing']}. "
                    "Available index levels: "
                    f"{['int', 'string', 'float']}"
                ),
            ),
            id="partial-missing-single",
        ),
        pytest.param(
            get_df_with_index_levels(["int", "string", "flat"]),
            ["missing", "int", "float"],
            pytest.raises(
                MissingIndexLevelsError,
                match=re.escape(
                    "The DataFrame is missing the following index levels: "
                    f"{['missing', 'float']}. "
                    "Available index levels: "
                    f"{['int', 'string', 'flat']}"
                ),
            ),
            id="partial-missing-multiple",
        ),
        pytest.param(
            get_df_with_index_levels(["int", 2010.0, 2020, 2025]),
            ["missing", "int", 2025, 2010.0, 2011.0, 2021],
            pytest.raises(
                MissingIndexLevelsError,
                match=re.escape(
                    "The DataFrame is missing the following index levels: "
                    f"{['missing', 2011.0, 2021]}. "
                    "Available index levels: "
                    f"{['int', 2010.0, 2020, 2025]}"
                ),
            ),
            id="missing-numerical-levels",
        ),
    ),
)
def test_assert_df_has_index_levels(inp, expected_levels, exp):
    with exp:
        assert_has_index_levels(inp, levels=expected_levels)
