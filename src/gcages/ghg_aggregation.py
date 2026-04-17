"""
Tools for calculating greenhouse gas aggregate timeseries
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from pandas_openscm.grouping import groupby_except
from pandas_openscm.index_manipulation import set_index_levels_func
from pandas_openscm.unit_conversion import convert_unit

from gcages.exceptions import MissingOptionalDependencyError
from gcages.renaming import SupportedNamingConventions, convert_variable_name

if TYPE_CHECKING:
    import pint

ALL_KYOTO_GHGS_GCAGES: tuple[str, ...] = (
    "Emissions|CO2|Fossil",
    "Emissions|CO2|Biosphere",
    "Emissions|CH4",
    "Emissions|N2O",
    "Emissions|HFC125",
    "Emissions|HFC134a",
    "Emissions|HFC143a",
    "Emissions|HFC152a",
    "Emissions|HFC227ea",
    "Emissions|HFC23",
    "Emissions|HFC236fa",
    "Emissions|HFC245fa",
    "Emissions|HFC32",
    "Emissions|HFC365mfc",
    "Emissions|HFC4310mee",
    "Emissions|NF3",
    "Emissions|SF6",
    "Emissions|C2F6",
    "Emissions|C3F8",
    "Emissions|C4F10",
    "Emissions|C5F12",
    "Emissions|C6F14",
    "Emissions|C7F16",
    "Emissions|C8F18",
    "Emissions|CF4",
    "Emissions|cC4F8",
)
"""
Emissions that are included when calculating aggreage Kyoto greenhouse gas emissions

These are specified in the gcages naming convention
(see [gcages.renaming.SupportedNamingConventions]).
"""


def calculate_kyoto_ghg(  # noqa: PLR0913
    indf: pd.DataFrame,
    indf_naming_convention: SupportedNamingConventions | None = None,
    kyoto_ghgs: tuple[str, ...] | None = None,
    gwp: str = "AR6GWP100",
    out_variable: str = "Kyoto GHG",
    out_unit: str = "MtCO2 / yr",
    ur: pint.facets.PlainRegistry | None = None,
    variable_level: str = "variable",
    unit_level: str = "unit",
) -> pd.DataFrame:
    """
    Calculate Kyoto greenhouse gas aggregate

    Parameters
    ----------
    indf
        Input data

    indf_naming_convention
        Naming convention used by `indf`

        Only required if `kyoto_ghgs` is `None`.
        If `kyoto_ghgs` is supplied, we assume that `indf` uses
        the same naming convention as the supplied `kyoto_ghgs`.
        If `kyoto_ghgs` is not supplied, we use `indf_naming_convention`
        to convert the list of Kyoto GHGs into the same naming convention
        as `indf`.

    kyoto_ghgs
        Gases to include in the aggregate

        If not supplied, we use
        [ALL_KYOTO_GHGS_GCAGES][gcages.ghg_aggregation.ALL_KYOTO_GHGS_GCAGES].
        See notes for `indf_naming_convention` to understand the implications
        of supplying or not supplying this variable for naming conventions.

    gwp
        GWP to use for calculating the aggregate

    out_variable
        Name to give the output variable

    out_unit
        Unit to use for the aggregation

        This must be some variation of t CO2 / yr.

    ur
        Unit registry to use for the unit conversions.

        If not supplied, we use [openscm_units.unit_registry][].

    variable_level
        Level in `indf`'s multi-index which contains variable names.

    unit_level
        Level in `indf`'s multi-index which contains unit information.

    Returns
    -------
    :
        Kyoto greenhouse gas aggregate timeseries
    """
    if kyoto_ghgs is None:
        if indf_naming_convention is None:
            msg = "If `kyoto_ghgs` is `None`, `indf_naming_convention` must be supplied"
            raise ValueError(msg)

        kyoto_ghgs_use = {
            convert_variable_name(
                v,
                from_convention=SupportedNamingConventions.GCAGES,
                to_convention=indf_naming_convention,
            )
            for v in ALL_KYOTO_GHGS_GCAGES
        }

    else:
        kyoto_ghgs_use = set(kyoto_ghgs)

    kyoto_ghgs_missing = kyoto_ghgs_use - set(
        indf.index.get_level_values(variable_level)
    )
    if kyoto_ghgs_missing:
        msg = (
            f"You are missing the following Kyoto GHGs: {kyoto_ghgs_missing}. "
            "Please either supply these gases "
            "or provide a different value for `kyoto_ghgs` to `calculate_kyoto_ghg`. "
            f"Currently {kyoto_ghgs=}."
        )
        raise AssertionError(msg)

    kyoto_ghgs_missing_by_group_l = []
    for meta, gdf in groupby_except(indf, [variable_level, unit_level]):
        kyoto_ghgs_missing_group = kyoto_ghgs_use - set(
            gdf.index.get_level_values(variable_level)
        )
        if kyoto_ghgs_missing_group:
            tmp = pd.Series(
                [", ".join(sorted(kyoto_ghgs_missing_group))],  # type: ignore # pandas-stubs confused
                index=pd.MultiIndex.from_tuples(
                    [meta],
                    names=indf.index.names.difference([variable_level, unit_level]),  # type: ignore # pandas-stubs confused
                ),
                name="missing_kyoto_ghgs",
            )
            kyoto_ghgs_missing_by_group_l.append(tmp)

    if kyoto_ghgs_missing_by_group_l:
        kyoto_ghgs_missing_by_group = pd.concat(
            kyoto_ghgs_missing_by_group_l
        ).to_frame()
        msg = (
            "For some groups, you are missing some Kyoto GHGs. "
            "Please either supply these gases for these groups "
            "or provide a different value for `kyoto_ghgs` to `calculate_kyoto_ghg`. "
            f"Currently {kyoto_ghgs=}. "
            "The groups and their missing Kyoto GHGs are:\n"
            f"{kyoto_ghgs_missing_by_group}"
        )
        raise AssertionError(msg)

    if ur is None:
        try:
            import openscm_units

            ur_use: pint.facets.PlainRegistry = openscm_units.unit_registry  # type: ignore # openscm_units type info incorrect?
        except ImportError:
            raise MissingOptionalDependencyError(  # noqa: TRY003
                "calculate_kyoto_ghgs(..., ur=None, ...)", "openscm_units"
            )
    else:
        ur_use = ur

    with ur_use.context(gwp):  # type: ignore # something wrong with pint typing
        components_same_unit = convert_unit(
            indf.loc[indf.index.get_level_values(variable_level).isin(kyoto_ghgs_use)],
            desired_units=out_unit,
            unit_level=unit_level,
            ur=ur_use,
        )

        res = set_index_levels_func(
            groupby_except(components_same_unit, variable_level).sum(),
            {variable_level: out_variable},
        )

    return res
