"""
Tests of `gcages.assertions.assert_metadata_values_all_allowed`
"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from gcages.assertions import (
    NotAllowedMetadataValuesError,
    assert_metadata_values_all_allowed,
)


def get_df(index):
    return pd.DataFrame(
        np.zeros((index.shape[0], 3)),
        columns=range(3),
        index=index,
    )


@pytest.mark.parametrize(
    "inp, metadata_key, allowed_values, exp",
    (
        pytest.param(
            get_df(
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "sa", "kg"),
                        ("vb", "sb", "kg"),
                    ],
                    names=["variable", "scenario", "unit"],
                )
            ),
            "variable",
            ["va", "vb"],
            does_not_raise(),
            id="all-allowed",
        ),
        pytest.param(
            get_df(
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "sa", "kg"),
                        ("vb", "sb", "kg"),
                    ],
                    names=["variable", "scenario", "unit"],
                )
            ),
            "variable",
            ["va", "vb", "vc"],
            does_not_raise(),
            id="all-allowed-superset",
        ),
        pytest.param(
            get_df(
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "sa", "kg"),
                        ("vb", "sb", "kg"),
                    ],
                    names=["variable", "scenario", "unit"],
                ),
            ),
            "variable",
            ["va", "vc"],
            pytest.raises(
                NotAllowedMetadataValuesError,
                match=re.escape(
                    f"The DataFrame contains disallowed values for {'variable'}: "
                    f"{['vb']}. "
                    f"Allowed values: {['va', 'vc']}"
                ),
            ),
            id="disallowed-values",
        ),
        pytest.param(
            get_df(
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "sa", "kg"),
                        ("vb", "sb", "kg"),
                        ("vd", "sb", "kg"),
                    ],
                    names=["variable", "scenario", "unit"],
                ),
            ),
            "variable",
            ["va", "vc"],
            pytest.raises(
                NotAllowedMetadataValuesError,
                match=re.escape(
                    f"The DataFrame contains disallowed values for {'variable'}: "
                    f"{['vb', 'vd']}. "
                    f"Allowed values: {['va', 'vc']}"
                ),
            ),
            id="disallowed-values-multiple",
        ),
        pytest.param(
            get_df(
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "sa", "kg", 0.0),
                        ("vb", "sb", "kg", 1.0),
                        ("vd", "sb", "kg", 2.0),
                        ("vd", "sb", "kg", 3.0),
                    ],
                    names=["variable", "scenario", "unit", "run_id"],
                ),
            ),
            "run_id",
            [0.0, 2.0, 4.0],
            pytest.raises(
                NotAllowedMetadataValuesError,
                match=re.escape(
                    f"The DataFrame contains disallowed values for {'run_id'}: "
                    f"{[1.0, 3.0]}. "
                    f"Allowed values: {[0.0, 2.0, 4.0]}"
                ),
            ),
            id="disallowed-values-multiple-float",
        ),
    ),
)
def test_assert_metadata_values_all_allowed(inp, metadata_key, allowed_values, exp):
    with exp:
        assert_metadata_values_all_allowed(
            inp, metadata_key=metadata_key, allowed_values=allowed_values
        )
