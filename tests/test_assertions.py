"""
Unit tests of `gcages.assertions`
"""

from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from gcages.assertions import assert_only_working_on_variable_unit_variations


@pytest.mark.parametrize(
    "indf, exp",
    (
        pytest.param(
            pd.DataFrame(
                np.arange(12).reshape((4, 3)),
                columns=[4, 5, 6],
                index=pd.MultiIndex.from_product(
                    [["a"], ["b", "c"], ["c", "d"]],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            does_not_raise(),
            id="only-variable-unit-variations",
        ),
        pytest.param(
            pd.DataFrame(
                np.arange(24).reshape((8, 3)),
                columns=[4, 5, 6],
                index=pd.MultiIndex.from_product(
                    [["sa", "sb"], ["ma"], ["b", "c"], ["c", "d"]],
                    names=["scenario", "model", "variable", "unit"],
                ),
            ),
            pytest.raises(
                AssertionError,
                match=re.escape("variations_in_other_cols=\nMultiIndex([('sa', 'ma'),"),
            ),
            id="scenario-variations",
        ),
    ),
)
def test_assert_only_working_on_variable_unit_variations(indf, exp):
    with exp:
        assert_only_working_on_variable_unit_variations(indf)
