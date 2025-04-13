"""
Unit tests of `gcages.testing`
"""

from __future__ import annotations

import re
import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gcages.exceptions import MissingOptionalDependencyError
from gcages.testing import (
    assert_frame_equal,
    get_ar6_harmonised_emissions,
    get_ar6_infilled_emissions,
    get_ar6_raw_emissions,
    get_key_testing_model_scenario_parameters,
)


@pytest.mark.parametrize(
    "to_call, args, exp_name, dependency, exp_dependency_name",
    (
        (
            get_key_testing_model_scenario_parameters,
            [],
            "get_key_testing_model_scenario_parameters",
            "pytest",
            "pytest",
        ),
        (
            get_ar6_raw_emissions,
            ["a", "b", "c"],
            "get_ar6_raw_emissions",
            "pandas_indexing.selectors",
            "pandas_indexing",
        ),
        (
            get_ar6_harmonised_emissions,
            ["a", "b", "c"],
            "get_ar6_harmonised_emissions",
            "pandas_indexing.selectors",
            "pandas_indexing",
        ),
        (
            get_ar6_infilled_emissions,
            ["a", "b", "c"],
            "get_ar6_infilled_emissions",
            "pandas_indexing.selectors",
            "pandas_indexing",
        ),
    ),
)
def test_missing_dependencies(to_call, args, exp_name, dependency, exp_dependency_name):
    with patch.dict(sys.modules, {dependency: None}):
        with pytest.raises(
            MissingOptionalDependencyError,
            match=(f"`{exp_name}` requires {exp_dependency_name} to be installed"),
        ):
            to_call(*args)


@pytest.mark.parametrize(
    "in_res, in_exp, exp",
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
            pd.DataFrame(
                np.arange(12).reshape((4, 3)),
                columns=[4, 5, 6],
                index=pd.MultiIndex.from_product(
                    [["a"], ["b", "c"], ["c", "d"]],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            does_not_raise(),
            id="passes",
        ),
        pytest.param(
            pd.DataFrame(
                np.arange(12).reshape((4, 3)),
                columns=[4, 5, 6],
                index=pd.MultiIndex.from_product(
                    [["sa"], ["b", "c"], ["c", "d"]],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            pd.DataFrame(
                np.arange(12).reshape((4, 3)),
                columns=[4, 5, 6],
                index=pd.MultiIndex.from_product(
                    [["sb"], ["b", "c"], ["c", "d"]],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "Differences in the scenario (res on the left): "
                    "idx_diffs=Index(['sa', 'sb'], dtype='object', name='scenario')"
                ),
            ),
            id="difference-in-scenario-name",
        ),
    ),
)
def test_assert_frame_equal(in_res, in_exp, exp):
    pytest.importorskip("pandas_indexing")
    with exp:
        assert_frame_equal(in_res, in_exp)
