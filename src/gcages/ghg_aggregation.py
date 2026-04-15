"""
Calculate aggregate Kyoto GHG emissions in CO2-equivalent units.
"""

import pandas as pd

from gcages.exceptions import MissingOptionalDependencyError

# from pandas_openscm.unit_conversion import convert_unit


def calculate_kyoto_ghgs(
    indf: pd.DataFrame, gwp: str = "AR6GWP100", reduced_ghg: bool = False
):
    """Calculate aggregate Kyoto GHG emissions in CO2-equivalent units."""
    try:
        import pandas_indexing as pix

        pix.set_openscm_registry_as_default()
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "calculate_kyoto_ghgs", requirement="pandas_indexing"
        ) from exc

    try:
        import pint

    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "calculate_kyoto_ghgs", requirement="pint"
        ) from exc

    if "Emissions|CO2" not in indf.pix.unique("variable"):
        res = (
            indf.loc[
                pix.isin(
                    variable=[
                        "Emissions|CO2|Biosphere",
                        "Emissions|CO2|Fossil",
                    ]
                )
            ]
            .groupby(["model", "scenario", "region", "unit"])
            .sum(min_count=2)
            .pix.assign(variable="Emissions|CO2")
        )
        indf = pix.concat(
            [
                indf,
                res,
            ]
        )

    kyoto_ghgs = [
        # 'Emissions|CO2|AFOLU',
        # 'Emissions|CO2|Energy and Industrial Processes',
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
    ]

    all_ghgs = [
        *kyoto_ghgs,
        "Emissions|CCl4",
        "Emissions|CFC11",
        "Emissions|CFC113",
        "Emissions|CFC114",
        "Emissions|CFC115",
        "Emissions|CFC12",
        "Emissions|CH2Cl2",
        "Emissions|CH3Br",
        "Emissions|CH3CCl3",
        "Emissions|CH3Cl",
        "Emissions|CHCl3",
        "Emissions|HCFC141b",
        "Emissions|HCFC142b",
        "Emissions|HCFC22",
        "Emissions|Halon1202",
        "Emissions|Halon1211",
        "Emissions|Halon1301",
        "Emissions|Halon2402",
        "Emissions|SO2F2",
    ]

    not_handled = set(indf.pix.unique("variable")) - set(kyoto_ghgs)

    problematic_strict = {
        "Emissions|OC",
        "Emissions|SOx",
        "Emissions|CO2|Biosphere",
        "Emissions|CO",
        "Emissions|NMVOC",
        "Emissions|BC",
        "Emissions|CO2|Fossil",
        "Emissions|NOx",
        "Emissions|NH3",
    }

    not_handled_problematic = not_handled - problematic_strict - set(all_ghgs)

    if not_handled_problematic:
        raise AssertionError(not_handled_problematic)

    if reduced_ghg:
        # climate-assessment calculate kyoto_ghgs with less gases
        climate_assessment_ghg = {
            "Emissions|HFC152a",
            "Emissions|HFC236fa",
            "Emissions|HFC245fa",
            "Emissions|HFC365mfc",
            "Emissions|NF3",
            "Emissions|C3F8",
            "Emissions|C4F10",
            "Emissions|C5F12",
            "Emissions|C7F16",
            "Emissions|C8F18",
            "Emissions|cC4F8",
        }
        kyoto_ghgs = [ghg for ghg in kyoto_ghgs if ghg not in climate_assessment_ghg]

    with pint.get_application_registry().context(gwp):
        gwp_str = f"{gwp[:3]}-{gwp[3:]}"

        res = (
            indf.loc[pix.isin(variable=kyoto_ghgs)]
            .pix.convert_unit("MtCO2 / yr")
            .groupby(["model", "scenario", "region", "unit"])
            # .sum(min_count=2)
            .sum()
            .pix.assign(
                variable=f"Emissions|Kyoto Gases ({gwp_str})", unit="Mt CO2-equiv/yr"
            )
        )
        # Huge headache here.
        # For some reason pandas_openscm.unit_conversion is not found!!
        # res = convert_unit(indf.loc[pix.isin(variable=kyoto_ghgs)],"MtCO2 / yr")
        # .groupby(["model", "scenario", "region", "unit"]).sum(min_count=2)
        # res = res.set_index(pd.Index(
        # [f"Emissions|Kyoto Gases ({gwp_str})"] * len(res), name="variable"
        # ), append=True)
        # .rename(index={"MtCO2 / yr": "Mt CO2-equiv/yr"}, level="unit")
        # res = res.reorder_levels(["model", "scenario", "region", "variable", "unit"])

    return res
