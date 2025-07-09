"""
Helpers for unit handling
"""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd
from pandas_indexing import assignlevel


def assert_has_no_pint_incompatible_characters(
    units: Collection[str], pint_incompatible_characters: Collection[str] | None = None
) -> None:
    """
    Assert that a collection does not contain pint-incompatible characters/strings

    Parameters
    ----------
    units
        Collection to check

        This is named `units` because we are normally checking collections of units

    pint_incompatible_characters
        Characters which are incompatible with pint

        This defaults to `{"-", "equiv"}`, which are commonly used in units,
        but not compatible with pint.

        You should not need to change this, but it is made an argument just in case

    Raises
    ------
    AssertionError
        `units` has elements that contain pint-incompatible characters
    """
    if pint_incompatible_characters is None:
        pint_incompatible_characters = {"-", "equiv"}

    unit_contains_pint_incompatible = [
        u for u in units if any(pi in u for pi in pint_incompatible_characters)
    ]
    if unit_contains_pint_incompatible:
        msg = (
            "The following units contain pint incompatible characters: "
            f"{unit_contains_pint_incompatible=}. "
            # Sort to make the error message deterministic
            f"pint_incompatible_characters={sorted(pint_incompatible_characters)}"
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
    return unit_str.replace("-", "").replace("equiv", "")


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
    # TODO: I'm not sure if this copy is necessary
    res = indf.copy()

    new_units = pd.Series(
        data=res.index.get_level_values(units_index_level).map(
            strip_pint_incompatible_characters_from_unit_string
        ),
        index=res.index,
        name=units_index_level,
    )

    return assignlevel(res, frame=new_units)
