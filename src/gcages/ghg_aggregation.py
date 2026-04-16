"""
Calculate aggregate Kyoto GHG emissions in CO2-equivalent units.
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


def calculate_kyoto_ghgs(
    indf: pd.DataFrame,
    indf_naming_convention: SupportedNamingConventions,
    gwp: str = "AR6GWP100",
    out_variable: str = "Emissions|Kyoto Gases",
    out_unit: str = "MtCO2 / yr",
    kyoto_ghgs: tuple[str, ...] | None = None,
    ur: pint.facets.PlainRegistry | None = None,
    variable_level: str = "variable",
    unit_level: str = "unit",
):
    if kyoto_ghgs is None:
        kyoto_ghgs_gcages = (
            "Emissions|CO2",
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
        kyoto_ghgs = [
            convert_variable_name(
                v,
                from_convention=SupportedNamingConventions.GCAGES,
                to_convention=indf_naming_convention,
            )
            for v in kyoto_ghgs_gcages
        ]

    kyoto_ghgs_missing = set(kyoto_ghgs) - set(
        indf.index.get_level_values("variable").unique()
    )
    if kyoto_ghgs_missing:
        msg = f"You are missing the following Kyoto GHGs: {kyoto_ghgs_missing}"
        raise AssertionError(msg)

    if ur is None:
        try:
            import openscm_units

            ur = openscm_units.unit_registry
        except ImportError:
            raise MissingOptionalDependencyError(  # noqa: TRY003
                "calculate_kyoto_ghgs(..., ur=None, ...)", "openscm_units"
            )

    with ur.context(gwp):
        components_same_unit = convert_unit(
            indf.loc[indf.index.get_level_values(variable_level).isin(kyoto_ghgs)],
            desired_units=out_unit,
            unit_level=unit_level,
            ur=ur,
        )
        res = set_index_levels_func(
            groupby_except(components_same_unit, variable_level).sum(),
            {variable_level: out_variable, unit_level: "Mt CO2eq/yr"},
        )
        # .pix.assign(variable=out_variable, unit="Mt CO2-equiv/yr")

    return res
