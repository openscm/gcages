"""
Helpers for unit handling
"""

from __future__ import annotations

import pandas as pd


def strip_pint_incompatible_characters_from_unit_string(unit_str: str) -> str:
    """
    Strip pint-incompatible characters from a unit string

    Parameters
    ----------
    unit_str
        Unit string from which to strip pint-incompatible characters

    Returns
    -------
    :
        `unit_str` with pint-incompatible characters removed
    """
    return unit_str.replace("-", "")


def strip_pint_incompatible_characters_from_units(
    indf: pd.DataFrame, units_index_level: str = "unit"
) -> pd.DataFrame:
    """
    Strip pint-incompatible characters from units

    Parameters
    ----------
    indf
        Input data from which to strip pint-incompatible characters

    units_index_level
        Column in `indf`'s index that holds the units values

    Returns
    -------
    :
        `indf` with pint-incompatible characters
        removed from the `units_index_level` of its index.
    """
    res = indf.copy()
    res.index = res.index.remove_unused_levels()
    res.index = res.index.set_levels(
        res.index.levels[res.index.names.index(units_index_level)].map(
            strip_pint_incompatible_characters_from_unit_string
        ),
        level=units_index_level,
    )

    return res
