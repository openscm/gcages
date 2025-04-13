"""
Tests of gcages.renaming for the AR6 CFC infilling database variables
"""

from __future__ import annotations

import pytest

from gcages.renaming import (
    SupportedNamingConventions,
    convert_variable_name,
)

cases_to_check_ar6_cfc_infilling_db = pytest.mark.parametrize(
    "ar6_cfc_infilling_db_variable, gcages_variable",
    tuple(
        pytest.param(
            ar6_cfc_infilling_db_variable,
            gcages_variable,
            id=gcages_variable,
        )
        for ar6_cfc_infilling_db_variable, gcages_variable in (
            ("Emissions|BC", "Emissions|BC"),
            ("Emissions|PFC|C2F6", "Emissions|C2F6"),
            ("Emissions|PFC|C3F8", "Emissions|C3F8"),
            ("Emissions|PFC|C4F10", "Emissions|C4F10"),
            ("Emissions|PFC|C5F12", "Emissions|C5F12"),
            ("Emissions|PFC|C6F14", "Emissions|C6F14"),
            ("Emissions|PFC|C7F16", "Emissions|C7F16"),
            ("Emissions|PFC|C8F18", "Emissions|C8F18"),
            ("Emissions|PFC|CF4", "Emissions|CF4"),
            ("Emissions|CH4", "Emissions|CH4"),
            ("Emissions|CO", "Emissions|CO"),
            ("Emissions|CO2", "Emissions|CO2"),
            ("Emissions|CO2|AFOLU", "Emissions|CO2|Biosphere"),
            (
                "Emissions|CO2|Energy and Industrial Processes",
                "Emissions|CO2|Fossil",
            ),
            ("Emissions|HFC|HFC125", "Emissions|HFC125"),
            ("Emissions|HFC|HFC134a", "Emissions|HFC134a"),
            ("Emissions|HFC|HFC143a", "Emissions|HFC143a"),
            ("Emissions|HFC|HFC152a", "Emissions|HFC152a"),
            ("Emissions|HFC|HFC227ea", "Emissions|HFC227ea"),
            ("Emissions|HFC|HFC23", "Emissions|HFC23"),
            ("Emissions|HFC|HFC236fa", "Emissions|HFC236fa"),
            ("Emissions|HFC|HFC245fa", "Emissions|HFC245fa"),
            ("Emissions|HFC|HFC32", "Emissions|HFC32"),
            ("Emissions|HFC|HFC365mfc", "Emissions|HFC365mfc"),
            ("Emissions|HFC|HFC43-10", "Emissions|HFC4310mee"),
            ("Emissions|CCl4", "Emissions|CCl4"),
            ("Emissions|CFC11", "Emissions|CFC11"),
            ("Emissions|CFC113", "Emissions|CFC113"),
            ("Emissions|CFC114", "Emissions|CFC114"),
            ("Emissions|CFC115", "Emissions|CFC115"),
            ("Emissions|CFC12", "Emissions|CFC12"),
            ("Emissions|CH2Cl2", "Emissions|CH2Cl2"),
            ("Emissions|CH3Br", "Emissions|CH3Br"),
            ("Emissions|CH3CCl3", "Emissions|CH3CCl3"),
            ("Emissions|CH3Cl", "Emissions|CH3Cl"),
            ("Emissions|CHCl3", "Emissions|CHCl3"),
            ("Emissions|HCFC141b", "Emissions|HCFC141b"),
            ("Emissions|HCFC142b", "Emissions|HCFC142b"),
            ("Emissions|HCFC22", "Emissions|HCFC22"),
            ("Emissions|Halon1202", "Emissions|Halon1202"),
            ("Emissions|Halon1211", "Emissions|Halon1211"),
            ("Emissions|Halon1301", "Emissions|Halon1301"),
            ("Emissions|Halon2402", "Emissions|Halon2402"),
            ("Emissions|N2O", "Emissions|N2O"),
            ("Emissions|NF3", "Emissions|NF3"),
            ("Emissions|NH3", "Emissions|NH3"),
            ("Emissions|NOx", "Emissions|NOx"),
            ("Emissions|OC", "Emissions|OC"),
            ("Emissions|SF6", "Emissions|SF6"),
            ("Emissions|SO2F2", "Emissions|SO2F2"),
            ("Emissions|Sulfur", "Emissions|SOx"),
            ("Emissions|VOC", "Emissions|NMVOC"),
            ("Emissions|PFC|cC4F8", "Emissions|cC4F8"),
        )
    ),
)


@cases_to_check_ar6_cfc_infilling_db
def test_convert_ar6_cfc_infilling_db_variable_to_gcages(
    ar6_cfc_infilling_db_variable, gcages_variable
):
    assert (
        convert_variable_name(
            ar6_cfc_infilling_db_variable,
            from_convention=SupportedNamingConventions.AR6_CFC_INFILLING_DB,
            to_convention=SupportedNamingConventions.GCAGES,
        )
        == gcages_variable
    )


@cases_to_check_ar6_cfc_infilling_db
def test_convert_ar6_cfc_infilling_db_variable_to_ar6_cfc_infilling_db(
    ar6_cfc_infilling_db_variable, gcages_variable
):
    assert (
        convert_variable_name(
            gcages_variable,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.AR6_CFC_INFILLING_DB,
        )
        == ar6_cfc_infilling_db_variable
    )
