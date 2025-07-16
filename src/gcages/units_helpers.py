"""
Helpers for unit handling
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

import pandas as pd
from pandas_openscm.index_manipulation import (
    set_index_levels_func,
)
from pandas_openscm.indexing import multi_index_match

from gcages.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    import pint


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


# TODO: move to pandas-openscm
def convert_unit_like(
    df: pd.DataFrame,
    target: pd.DataFrame,
    df_unit_level: str = "unit",
    target_unit_level: str | None = None,
    ur: pint.UnitRegistry | None = None,
) -> pd.DataFrame:
    """
    Convert to units like a different [pd.DataFrame][pandas.DataFrame]

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] of which to convert the units

    target
        [pd.DataFrame][pandas.DataFrame] whose units the result should match

    df_unit_level
        Level in `df`'s index which has unit information

    target_unit_level
        Level in `target`'s index which has unit information

        If not provided, we assume this is the same as `df_unit_level`

    ur
        Unit registry to use for determining unit conversions

        If not provided, we use [openscm_units.unit_registry][]

    Returns
    -------
    :
        `df` with units that match `target`

    Raises
    ------
    MissingOptionalDependencyError
        `ur` is `None` and openscm-units is not installed
    """
    if target_unit_level is None:
        target_unit_col_use = df_unit_level
    else:
        target_unit_col_use = target_unit_level

    extra_index_levels_target = target.index.names.difference(df.index.names)  # type: ignore # pandas-stubs confused
    if extra_index_levels_target:
        msg = (
            "Haven't worked out the logic "
            "when the target has index levels which aren't in `df`. "
            f"{extra_index_levels_target=}"
        )
        raise NotImplementedError(msg)

    df_units_s = df.index.get_level_values(df_unit_level).to_series(
        index=df.index.droplevel(df_unit_level), name="df_unit"
    )
    target_units_s = target.index.get_level_values(target_unit_col_use).to_series(
        index=target.index.droplevel(target_unit_col_use), name="target_unit"
    )
    unit_map = pd.DataFrame([*target_units_s.align(df_units_s)]).T

    if (unit_map["df_unit"] == unit_map["target_unit"]).all():
        # Already in matching units
        return df

    if ur is None:
        try:
            import openscm_units

            ur = openscm_units.unit_registry
        except ImportError:
            raise MissingOptionalDependencyError(  # noqa: TRY003
                "convert_unit_like(..., ur=None, ...)", "openscm_units"
            )

    df_converted = df.reset_index(df_unit_level, drop=True)
    for (df_unit, target_unit), conversion_df in unit_map.groupby(
        ["df_unit", "target_unit"]
    ):
        conversion_factor = ur(df_unit).to(target_unit).m
        to_alter_loc = multi_index_match(df_converted.index, conversion_df.index)  # type: ignore
        df_converted.loc[to_alter_loc, :] *= conversion_factor

    # All conversions done so can simply assign the unit column.
    # When moving to pandas-openscm, check this carefully
    # using different input row order etc.

    unit_map_reordered = unit_map.reorder_levels(df_converted.index.names)
    res = set_index_levels_func(
        df_converted,
        {df_unit_level: unit_map_reordered["target_unit"].loc[df_converted.index]},
    ).reorder_levels(df.index.names)

    return res
