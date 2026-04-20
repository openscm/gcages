"""
Tests of gcages.renaming
"""

from __future__ import annotations

import itertools
import re

import pandas as pd
import pytest

from gcages.exceptions import UnrecognisedValueError
from gcages.renaming import (
    SupportedNamingConventions,
    convert_variable_name,
    rename_variables,
)


@pytest.mark.parametrize("from_nc", [v for v in SupportedNamingConventions])
def test_convert_variable_name_unknown_error(from_nc):
    with pytest.raises(
        UnrecognisedValueError,
        match=re.escape(
            f"'Emissions|junk' is not a recognised value for the {from_nc} "
            "naming convention. "
            "Did you mean 'Emissions|OC' or 'Emissions|CO' or 'Emissions|BC'? "
            "The full list of known values is:"
        ),
    ):
        convert_variable_name(
            "Emissions|junk",
            from_convention=from_nc,
            to_convention=SupportedNamingConventions.GCAGES,
        )


naming_convention_combos = pytest.mark.parametrize(
    "from_nc, to_nc",
    tuple(
        [
            pytest.param(
                from_nc,
                to_nc,
                id=f"{from_nc.value}-{to_nc.value}",
            )
            for from_nc, to_nc in itertools.combinations(
                [v for v in SupportedNamingConventions], 2
            )
        ]
    ),
)


@pytest.mark.parametrize(
    "gcages_variable",
    (
        "Emissions|BC",
        "Emissions|C2F6",
        "Emissions|C3F8",
        "Emissions|C4F10",
        "Emissions|C5F12",
        "Emissions|C6F14",
        "Emissions|C7F16",
        "Emissions|C8F18",
        "Emissions|CF4",
        "Emissions|CH4",
        "Emissions|CO",
        "Emissions|CO2",
        "Emissions|CO2|Biosphere",
        "Emissions|CO2|Fossil",
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
        "Emissions|N2O",
        "Emissions|NF3",
        "Emissions|NH3",
        "Emissions|NOx",
        "Emissions|OC",
        "Emissions|SF6",
        "Emissions|SO2F2",
        "Emissions|SOx",
        "Emissions|NMVOC",
        "Emissions|cC4F8",
    ),
)
@naming_convention_combos
def test_all_combos(gcages_variable, from_nc, to_nc):
    # Get starting point
    start = convert_variable_name(
        gcages_variable,
        from_convention=SupportedNamingConventions.GCAGES,
        to_convention=from_nc,
    )

    # Check the round trip
    assert (
        convert_variable_name(
            convert_variable_name(start, from_convention=from_nc, to_convention=to_nc),
            from_convention=to_nc,
            to_convention=from_nc,
        )
        == start
    )


@pytest.mark.parametrize("to_nc", SupportedNamingConventions)
@pytest.mark.parametrize(
    "index_level_in, index_level_exp",
    (
        (None, "variable"),
        ("variables", "variables"),
    ),
)
@pytest.mark.parametrize(
    "copy, copy_exp",
    (
        (None, True),
        (False, False),
    ),
)
def test_rename_variables(to_nc, index_level_in, index_level_exp, copy, copy_exp):
    call_kwargs = {}

    from_nc = SupportedNamingConventions.GCAGES
    gcages_variables = [
        "Emissions|CO2|Biosphere",
        "Emissions|CO2|Fossil",
        "Emissions|HFC4310mee",
        "Emissions|SOx",
        "Emissions|NMVOC",
    ]

    tmp = pd.DataFrame(
        range(len(gcages_variables)),
        pd.MultiIndex.from_tuples(
            [(v, "t") for v in gcages_variables], names=["variable", "unit"]
        ),
    )

    if index_level_in is not None:
        call_kwargs["index_level"] = index_level_in
        tmp = tmp.rename_axis(index={"variable": index_level_in})

    if copy is not None:
        call_kwargs["copy"] = copy

    res = rename_variables(
        tmp, from_convention=from_nc, to_convention=to_nc, **call_kwargs
    )

    if copy_exp:
        assert id(res) != id(tmp)
    else:
        assert id(res) == id(tmp)

    assert list(res.index.get_level_values(index_level_exp)) == [
        convert_variable_name(
            v,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=to_nc,
        )
        for v in gcages_variables
    ]

    res_roundtrip = rename_variables(
        res, from_convention=to_nc, to_convention=from_nc, **call_kwargs
    )

    pd.testing.assert_frame_equal(tmp, res_roundtrip)
