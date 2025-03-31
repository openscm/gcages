"""
Unit tests of `gcages.ar6.infilling`
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from gcages.ar6.infilling import infill_scenario, load_ar6_infilling_db
from gcages.exceptions import MissingOptionalDependencyError


@pytest.mark.parametrize("cfcs", (True, False))
def test_load_ar6_infilling_db_wrong_sha(cfcs):
    with pytest.raises(AssertionError):
        load_ar6_infilling_db(__file__, cfcs=cfcs)


@pytest.mark.parametrize(
    "to_call, args, exp_name, dependency, exp_dependency_name",
    (
        (
            infill_scenario,
            ["indf", "infillers"],
            "infill_scenario",
            "pandas_indexing.core",
            "pandas_indexing",
        ),
        (
            infill_scenario,
            ["indf", "infillers"],
            "infill_scenario",
            "pyam",
            "pyam",
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
