"""
Emissions variables database
"""

from __future__ import annotations

import pandas as pd

EMISSIONS_VARIABLES = pd.DataFrame(
    [
        ("Emissions|BC", "Emissions|BC", "Emissions|BC"),
        ("Emissions|C2F6", "Emissions|C2F6", "Emissions|PFC|C2F6"),
        ("Emissions|C3F8", "Emissions|C3F8", "Emissions|PFC|C3F8"),
        ("Emissions|C4F10", "Emissions|C4F10", "Emissions|PFC|C4F10"),
        ("Emissions|C5F12", "Emissions|C5F12", "Emissions|PFC|C5F12"),
        ("Emissions|C6F14", "Emissions|C6F14", "Emissions|PFC|C6F14"),
        ("Emissions|C7F16", "Emissions|C7F16", "Emissions|PFC|C7F16"),
        ("Emissions|C8F18", "Emissions|C8F18", "Emissions|PFC|C8F18"),
        ("Emissions|CF4", "Emissions|CF4", "Emissions|PFC|CF4"),
        ("Emissions|CH4", "Emissions|CH4", "Emissions|CH4"),
        ("Emissions|CO", "Emissions|CO", "Emissions|CO"),
        ("Emissions|CO2", "Emissions|CO2", "Emissions|CO2"),
        (
            "Emissions|CO2|Biosphere",
            "Emissions|CO2|MAGICC AFOLU",
            "Emissions|CO2|AFOLU",
        ),
        (
            "Emissions|CO2|Fossil",
            "Emissions|CO2|MAGICC Fossil and Industrial",
            "Emissions|CO2|Energy and Industrial Processes",
        ),
        ("Emissions|HFC125", "Emissions|HFC125", "Emissions|HFC|HFC125"),
        ("Emissions|HFC134a", "Emissions|HFC134a", "Emissions|HFC|HFC134a"),
        ("Emissions|HFC143a", "Emissions|HFC143a", "Emissions|HFC|HFC143a"),
        ("Emissions|HFC152a", "Emissions|HFC152a", "Emissions|HFC|HFC152a"),
        ("Emissions|HFC227ea", "Emissions|HFC227ea", "Emissions|HFC|HFC227ea"),
        ("Emissions|HFC23", "Emissions|HFC23", "Emissions|HFC|HFC23"),
        ("Emissions|HFC236fa", "Emissions|HFC236fa", "Emissions|HFC|HFC236fa"),
        ("Emissions|HFC245fa", "Emissions|HFC245fa", "Emissions|HFC|HFC245fa"),
        ("Emissions|HFC32", "Emissions|HFC32", "Emissions|HFC|HFC32"),
        ("Emissions|HFC365mfc", "Emissions|HFC365mfc", "Emissions|HFC|HFC365mfc"),
        ("Emissions|HFC4310mee", "Emissions|HFC4310mee", "Emissions|HFC|HFC43-10"),
        (
            "Emissions|CCl4",
            "Emissions|CCl4",
            "Emissions|Montreal Gases|CCl4",
        ),
        (
            "Emissions|CFC11",
            "Emissions|CFC11",
            "Emissions|Montreal Gases|CFC|CFC11",
        ),
        (
            "Emissions|CFC113",
            "Emissions|CFC113",
            "Emissions|Montreal Gases|CFC|CFC113",
        ),
        (
            "Emissions|CFC114",
            "Emissions|CFC114",
            "Emissions|Montreal Gases|CFC|CFC114",
        ),
        (
            "Emissions|CFC115",
            "Emissions|CFC115",
            "Emissions|Montreal Gases|CFC|CFC115",
        ),
        (
            "Emissions|CFC12",
            "Emissions|CFC12",
            "Emissions|Montreal Gases|CFC|CFC12",
        ),
        (
            "Emissions|CH2Cl2",
            "Emissions|CH2Cl2",
            "Emissions|Montreal Gases|CH2Cl2",
        ),
        (
            "Emissions|CH3Br",
            "Emissions|CH3Br",
            "Emissions|Montreal Gases|CH3Br",
        ),
        (
            "Emissions|CH3CCl3",
            "Emissions|CH3CCl3",
            "Emissions|Montreal Gases|CH3CCl3",
        ),
        (
            "Emissions|CH3Cl",
            "Emissions|CH3Cl",
            "Emissions|Montreal Gases|CH3Cl",
        ),
        (
            "Emissions|CHCl3",
            "Emissions|CHCl3",
            "Emissions|Montreal Gases|CHCl3",
        ),
        (
            "Emissions|HCFC141b",
            "Emissions|HCFC141b",
            "Emissions|Montreal Gases|HCFC141b",
        ),
        (
            "Emissions|HCFC142b",
            "Emissions|HCFC142b",
            "Emissions|Montreal Gases|HCFC142b",
        ),
        (
            "Emissions|HCFC22",
            "Emissions|HCFC22",
            "Emissions|Montreal Gases|HCFC22",
        ),
        (
            "Emissions|Halon1202",
            "Emissions|Halon1202",
            "Emissions|Montreal Gases|Halon1202",
        ),
        (
            "Emissions|Halon1211",
            "Emissions|Halon1211",
            "Emissions|Montreal Gases|Halon1211",
        ),
        (
            "Emissions|Halon1301",
            "Emissions|Halon1301",
            "Emissions|Montreal Gases|Halon1301",
        ),
        (
            "Emissions|Halon2402",
            "Emissions|Halon2402",
            "Emissions|Montreal Gases|Halon2402",
        ),
        ("Emissions|N2O", "Emissions|N2O", "Emissions|N2O"),
        ("Emissions|NF3", "Emissions|NF3", "Emissions|NF3"),
        ("Emissions|NH3", "Emissions|NH3", "Emissions|NH3"),
        ("Emissions|NOx", "Emissions|NOx", "Emissions|NOx"),
        ("Emissions|OC", "Emissions|OC", "Emissions|OC"),
        ("Emissions|SF6", "Emissions|SF6", "Emissions|SF6"),
        ("Emissions|SO2F2", "Emissions|SO2F2", "Emissions|SO2F2"),
        ("Emissions|SOx", "Emissions|Sulfur", "Emissions|Sulfur"),
        ("Emissions|NMVOC", "Emissions|VOC", "Emissions|VOC"),
        ("Emissions|cC4F8", "Emissions|cC4F8", "Emissions|PFC|cC4F8"),
    ],
    columns=["gcages", "openscm_runner", "iamc"],
)
"""
Database of emissions variables names according to different naming schemes

You will likely not need to access this variable directly,
and instead will use one of:

- [convert_gcages_variable_to_iamc][gcages.renaming.].
- [convert_gcages_variable_to_openscm_runner][gcages.renaming.].
- [convert_iamc_variable_to_gcages][gcages.renaming.].
- [convert_iamc_variable_to_openscm_runner][gcages.renaming.].
- [convert_openscm_runner_variable_to_gcages][gcages.renaming.].
- [convert_openscm_runner_variable_to_iamc][gcages.renaming.].
"""
