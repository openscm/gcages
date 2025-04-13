"""
Tests of gcages.renaming for RCMIP variables
"""

from __future__ import annotations

import pytest

from gcages.renaming import (
    SupportedNamingConventions,
    convert_variable_name,
)

cases_to_check_rcmip = pytest.mark.parametrize(
    "rcmip_variable, gcages_variable",
    tuple(
        pytest.param(
            rcmip_variable,
            gcages_variable,
            id=gcages_variable,
        )
        for rcmip_variable, gcages_variable in (
            ("Emissions|BC", "Emissions|BC"),
            ("Emissions|F-Gases|PFC|C2F6", "Emissions|C2F6"),
            ("Emissions|F-Gases|PFC|C3F8", "Emissions|C3F8"),
            ("Emissions|F-Gases|PFC|C4F10", "Emissions|C4F10"),
            ("Emissions|F-Gases|PFC|C5F12", "Emissions|C5F12"),
            ("Emissions|F-Gases|PFC|C6F14", "Emissions|C6F14"),
            ("Emissions|F-Gases|PFC|C7F16", "Emissions|C7F16"),
            ("Emissions|F-Gases|PFC|C8F18", "Emissions|C8F18"),
            ("Emissions|F-Gases|PFC|CF4", "Emissions|CF4"),
            ("Emissions|CH4", "Emissions|CH4"),
            ("Emissions|CO", "Emissions|CO"),
            ("Emissions|CO2", "Emissions|CO2"),
            ("Emissions|CO2|MAGICC AFOLU", "Emissions|CO2|Biosphere"),
            (
                "Emissions|CO2|MAGICC Fossil and Industrial",
                "Emissions|CO2|Fossil",
            ),
            ("Emissions|F-Gases|HFC|HFC125", "Emissions|HFC125"),
            ("Emissions|F-Gases|HFC|HFC134a", "Emissions|HFC134a"),
            ("Emissions|F-Gases|HFC|HFC143a", "Emissions|HFC143a"),
            ("Emissions|F-Gases|HFC|HFC152a", "Emissions|HFC152a"),
            ("Emissions|F-Gases|HFC|HFC227ea", "Emissions|HFC227ea"),
            ("Emissions|F-Gases|HFC|HFC23", "Emissions|HFC23"),
            ("Emissions|F-Gases|HFC|HFC236fa", "Emissions|HFC236fa"),
            ("Emissions|F-Gases|HFC|HFC245fa", "Emissions|HFC245fa"),
            ("Emissions|F-Gases|HFC|HFC32", "Emissions|HFC32"),
            ("Emissions|F-Gases|HFC|HFC365mfc", "Emissions|HFC365mfc"),
            ("Emissions|F-Gases|HFC|HFC4310mee", "Emissions|HFC4310mee"),
            ("Emissions|Montreal Gases|CCl4", "Emissions|CCl4"),
            ("Emissions|Montreal Gases|CFC|CFC11", "Emissions|CFC11"),
            ("Emissions|Montreal Gases|CFC|CFC113", "Emissions|CFC113"),
            ("Emissions|Montreal Gases|CFC|CFC114", "Emissions|CFC114"),
            ("Emissions|Montreal Gases|CFC|CFC115", "Emissions|CFC115"),
            ("Emissions|Montreal Gases|CFC|CFC12", "Emissions|CFC12"),
            ("Emissions|Montreal Gases|CH2Cl2", "Emissions|CH2Cl2"),
            ("Emissions|Montreal Gases|CH3Br", "Emissions|CH3Br"),
            ("Emissions|Montreal Gases|CH3CCl3", "Emissions|CH3CCl3"),
            ("Emissions|Montreal Gases|CH3Cl", "Emissions|CH3Cl"),
            ("Emissions|Montreal Gases|CHCl3", "Emissions|CHCl3"),
            ("Emissions|Montreal Gases|HCFC141b", "Emissions|HCFC141b"),
            ("Emissions|Montreal Gases|HCFC142b", "Emissions|HCFC142b"),
            ("Emissions|Montreal Gases|HCFC22", "Emissions|HCFC22"),
            ("Emissions|Montreal Gases|Halon1202", "Emissions|Halon1202"),
            ("Emissions|Montreal Gases|Halon1211", "Emissions|Halon1211"),
            ("Emissions|Montreal Gases|Halon1301", "Emissions|Halon1301"),
            ("Emissions|Montreal Gases|Halon2402", "Emissions|Halon2402"),
            ("Emissions|N2O", "Emissions|N2O"),
            ("Emissions|F-Gases|NF3", "Emissions|NF3"),
            ("Emissions|NH3", "Emissions|NH3"),
            ("Emissions|NOx", "Emissions|NOx"),
            ("Emissions|OC", "Emissions|OC"),
            ("Emissions|F-Gases|SF6", "Emissions|SF6"),
            ("Emissions|F-Gases|SO2F2", "Emissions|SO2F2"),
            ("Emissions|Sulfur", "Emissions|SOx"),
            ("Emissions|VOC", "Emissions|NMVOC"),
            ("Emissions|F-Gases|PFC|cC4F8", "Emissions|cC4F8"),
        )
    ),
)


@cases_to_check_rcmip
def test_convert_rcmip_variable_to_gcages(rcmip_variable, gcages_variable):
    assert (
        convert_variable_name(
            rcmip_variable,
            from_convention=SupportedNamingConventions.RCMIP,
            to_convention=SupportedNamingConventions.GCAGES,
        )
        == gcages_variable
    )


@cases_to_check_rcmip
def test_convert_rcmip_variable_to_rcmip(rcmip_variable, gcages_variable):
    assert (
        convert_variable_name(
            gcages_variable,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.RCMIP,
        )
        == rcmip_variable
    )
