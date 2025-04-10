"""
Tests of gcages.renaming
"""

from __future__ import annotations

import pytest

from gcages.renaming import (
    convert_gcages_variable_to_iamc,
    convert_iamc_variable_to_gcages,
)

cases_to_check = pytest.mark.parametrize(
    "iamc_variable, gcages_variable",
    tuple(
        pytest.param(
            iamc_variable,
            gcages_variable,
            id=gcages_variable,
        )
        for iamc_variable, gcages_variable in {
            "Emissions|BC": "Emissions|BC",
            "Emissions|C2F6": "Emissions|C2F6",
            "Emissions|C3F8": "Emissions|C3F8",
            "Emissions|C4F10": "Emissions|C4F10",
            "Emissions|C5F12": "Emissions|C5F12",
            "Emissions|C6F14": "Emissions|C6F14",
            "Emissions|C7F16": "Emissions|C7F16",
            "Emissions|C8F18": "Emissions|C8F18",
            "Emissions|CF4": "Emissions|CF4",
            "Emissions|CH4": "Emissions|CH4",
            "Emissions|CO": "Emissions|CO",
            "Emissions|CO2|AFOLU": "Emissions|CO2|AFOLU",
            "Emissions|CO2|Energy and Industrial Processes": "Emissions|CO2|Energy and Industrial Processes",  # noqa: E501
            "Emissions|HFC|HFC125": "Emissions|HFC125",
            "Emissions|HFC|HFC134a": "Emissions|HFC134a",
            "Emissions|HFC|HFC143a": "Emissions|HFC143a",
            "Emissions|HFC|HFC152a": "Emissions|HFC152a",
            "Emissions|HFC|HFC227ea": "Emissions|HFC227ea",
            "Emissions|HFC|HFC23": "Emissions|HFC23",
            "Emissions|HFC|HFC236fa": "Emissions|HFC236fa",
            "Emissions|HFC|HFC245fa": "Emissions|HFC245fa",
            "Emissions|HFC|HFC32": "Emissions|HFC32",
            "Emissions|HFC|HFC365mfc": "Emissions|HFC365mfc",
            "Emissions|HFC|HFC43-10": "Emissions|HFC4310mee",
            "Emissions|Montreal Gases|CCl4": "Emissions|CCl4",
            "Emissions|Montreal Gases|CFC|CFC11": "Emissions|CFC11",
            "Emissions|Montreal Gases|CFC|CFC113": "Emissions|CFC113",
            "Emissions|Montreal Gases|CFC|CFC114": "Emissions|CFC114",
            "Emissions|Montreal Gases|CFC|CFC115": "Emissions|CFC115",
            "Emissions|Montreal Gases|CFC|CFC12": "Emissions|CFC12",
            "Emissions|Montreal Gases|CH2Cl2": "Emissions|CH2Cl2",
            "Emissions|Montreal Gases|CH3Br": "Emissions|CH3Br",
            "Emissions|Montreal Gases|CH3CCl3": "Emissions|CH3CCl3",
            "Emissions|Montreal Gases|CH3Cl": "Emissions|CH3Cl",
            "Emissions|Montreal Gases|CHCl3": "Emissions|CHCl3",
            "Emissions|Montreal Gases|HCFC141b": "Emissions|HCFC141b",
            "Emissions|Montreal Gases|HCFC142b": "Emissions|HCFC142b",
            "Emissions|Montreal Gases|HCFC22": "Emissions|HCFC22",
            "Emissions|Montreal Gases|Halon1202": "Emissions|Halon1202",
            "Emissions|Montreal Gases|Halon1211": "Emissions|Halon1211",
            "Emissions|Montreal Gases|Halon1301": "Emissions|Halon1301",
            "Emissions|Montreal Gases|Halon2402": "Emissions|Halon2402",
            "Emissions|N2O": "Emissions|N2O",
            "Emissions|NF3": "Emissions|NF3",
            "Emissions|NH3": "Emissions|NH3",
            "Emissions|NOx": "Emissions|NOx",
            "Emissions|OC": "Emissions|OC",
            "Emissions|SF6": "Emissions|SF6",
            "Emissions|SO2F2": "Emissions|SO2F2",
            "Emissions|Sulfur": "Emissions|SOx",
            "Emissions|VOC": "Emissions|NMVOC",
            "Emissions|cC4F8": "Emissions|cC4F8",
            "junk": "junk",
        }.items()
    ),
)


@cases_to_check
def test_convert_iamc_variable_to_gcages(iamc_variable, gcages_variable):
    assert convert_iamc_variable_to_gcages(iamc_variable) == gcages_variable


@cases_to_check
def test_convert_gcages_variable_to_iamc(iamc_variable, gcages_variable):
    assert convert_gcages_variable_to_iamc(gcages_variable) == iamc_variable


@cases_to_check
def test_round_tripping(iamc_variable, gcages_variable):
    assert (
        convert_gcages_variable_to_iamc(convert_iamc_variable_to_gcages(iamc_variable))
        == iamc_variable
    )
    assert (
        convert_iamc_variable_to_gcages(
            convert_gcages_variable_to_iamc(gcages_variable)
        )
        == gcages_variable
    )
