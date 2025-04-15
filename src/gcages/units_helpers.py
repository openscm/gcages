"""
Helpers for unit handling
"""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd


def assert_has_no_pint_incompatible_characters(
    units: Collection[str], pint_incompatible_characters: Collection[str] = {"-"}
) -> None:
    """
    Assert that a collection does not contain pint-incompatible characters

    Parameters
    ----------
    units
        Collection to check

        This is named `units` because we are normally checking collections of units

    pint_incompatible_characters
        Characters which are incompatible with pint

        You should not need to change this, but it is made an argument just in case

    Raises
    ------
    AssertionError
        `units` has elements that contain pint-incompatible characters
    """
    unit_contains_pint_incompatible = [
        u for u in units if any(pi in u for pi in pint_incompatible_characters)
    ]
    if unit_contains_pint_incompatible:
        msg = (
            "The following units contain pint incompatible characters: "
            f"{unit_contains_pint_incompatible=}. "
            f"{pint_incompatible_characters=}"
        )
        raise AssertionError(msg)


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
    res.index = res.index.remove_unused_levels()  # type: ignore # not in pandas-stubs
    res.index = res.index.set_levels(  # type: ignore # pandas-stubs out of date
        res.index.levels[res.index.names.index(units_index_level)].map(  # type: ignore # pandas-stubs out of date
            strip_pint_incompatible_characters_from_unit_string
        ),
        level=units_index_level,
    )

    return res
