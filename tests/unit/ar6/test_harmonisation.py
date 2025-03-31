"""
Unit tests of `gcages.ar6.harmonisation`
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from gcages.ar6.harmonisation import harmonise_scenario, load_ar6_historical_emissions
from gcages.exceptions import MissingOptionalDependencyError


def test_load_ar6_infilling_db_wrong_sha():
    with pytest.raises(AssertionError):
        load_ar6_historical_emissions(__file__)


@pytest.mark.parametrize(
    "to_call, args, exp_name, dependency, exp_dependency_name",
    (
        (
            harmonise_scenario,
            ["indf", "history", "year", "overrides", "calc_scaling_year"],
            "harmonise_scenario",
            "scipy",
            "scipy",
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
