"""
Unit tests of `gcages.units_helpers`
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from gcages.units_helpers import (
    assert_has_no_pint_incompatible_characters,
    strip_pint_incompatible_characters_from_units,
)


@pytest.mark.parametrize(
    "units_index_level, units_index_level_exp",
    (
        (None, "unit"),
        ("unit", "unit"),
        ("units", "units,"),
    ),
)
def test_strip_pint_incompatible_characters_from_units(
    units_index_level, units_index_level_exp
):
    kwargs = {}
    if units_index_level is not None:
        units_index_level_exp = "unit"

    start = pd.DataFrame(
        np.arange(18).reshape((6, 3)),
        columns=[2010, 2020, 2050],
        index=pd.MultiIndex.from_tuples(
            [
                ("sa", "CO2", "MtCO2 / yr"),
                ("sb", "CO2", "MtCO2 / yr"),
                ("sa", "hfc4310-mee", "MtHFC4310-mee / yr"),
                ("sb", "hfc4310-mee", "MtHFC4310-mee / yr"),
                ("sa", "hfc-125", "kt HFC-125/yr"),
                ("sb", "hfc-125", "kt HFC-125/yr"),
            ],
            names=["scenario", "variable", units_index_level_exp],
        ),
    )

    res = strip_pint_incompatible_characters_from_units(start, **kwargs)

    exp_units = {"MtCO2 / yr", "MtHFC4310mee / yr", "kt HFC125/yr"}
    assert (
        set(res.index.get_level_values(units_index_level_exp).unique().tolist())
        == exp_units
    )


def test_assert_has_no_pint_incompatible_characters():
    error_msg = re.escape(
        "The following units contain pint incompatible characters: "
        "unit_contains_pint_incompatible=['Mt HFC43-10/yr']. "
        "pint_incompatible_characters={'-'}"
    )
    with pytest.raises(AssertionError, match=error_msg):
        assert_has_no_pint_incompatible_characters(
            ["Mt CO2/yr", "Mt CH4/yr", "Mt HFC43-10/yr", "Mt HFC23/yr"]
        )
