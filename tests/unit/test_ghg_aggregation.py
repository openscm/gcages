"""
Tests of the `gcages.ghg_aggregation` using ar6 data.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pandas as pd
import pandas_openscm.io
import pandas_openscm.testing
import pytest
from pandas_openscm.grouping import groupby_except
from pandas_openscm.index_manipulation import set_index_levels_func

from gcages.ghg_aggregation import calculate_kyoto_ghg
from gcages.renaming import SupportedNamingConventions, rename_variables

openscm_units = pytest.importorskip("openscm_units")
pint = pytest.importorskip("pint")

CMIP7_SCENARIOMIP_OUT_DIR = (
    Path(__file__).parents[1]
    / "regression/cmip7-scenariomip"
    / "cmip7-scenariomip-output"
)


@pytest.fixture(scope="module")
def indf_basic():
    res = pd.DataFrame(
        [
            [100, 110, 120],
            [10, 11, 12],
            [1000.0, 2000.0, 3000.0],
            [200, 100, 300],
            [5, 6, 7],
            [1000.0, 500.0, 0.0],
        ],
        columns=[2010, 2020, 2025],
        index=pd.MultiIndex.from_tuples(
            [
                ("a", "CO2", "MtCO2 / yr"),
                ("a", "CH4", "MtCH4 / yr"),
                ("a", "N2O", "ktN2O / yr"),
                ("b", "CO2", "MtCO2 / yr"),
                ("b", "CH4", "MtCH4 / yr"),
                ("b", "N2O", "ktN2O / yr"),
            ],
            names=["ms", "variable", "unit"],
        ),
    )

    return res


def test_calculate_kyoto_ghg_basic(indf_basic):
    res = calculate_kyoto_ghg(
        indf_basic,
        kyoto_ghgs=("CO2", "CH4", "N2O"),
    )

    exp = pd.DataFrame(
        [
            [
                100 + 10 * 27.9 + 1.0 * 273.0,
                110 + 11 * 27.9 + 2.0 * 273.0,
                120 + 12 * 27.9 + 3.0 * 273.0,
            ],
            [
                200 + 5 * 27.9 + 1.0 * 273.0,
                100 + 6 * 27.9 + 0.5 * 273.0,
                300 + 7 * 27.9 + 0.0 * 273.0,
            ],
        ],
        columns=[2010, 2020, 2025],
        index=pd.MultiIndex.from_tuples(
            [
                ("a", "Kyoto GHG", "MtCO2 / yr"),
                ("b", "Kyoto GHG", "MtCO2 / yr"),
            ],
            names=["ms", "variable", "unit"],
        ),
    )

    pandas_openscm.testing.assert_frame_alike(res, exp)


def test_calculate_kyoto_ghg_all_missing(indf_basic):
    indf = indf_basic.loc[~(indf_basic.index.get_level_values("variable") == "N2O")]
    error_msg = re.escape(
        "You are missing the following Kyoto GHGs: {'N2O'}. "
        "Please either supply these gases "
        "or provide a different value for `kyoto_ghgs` to `calculate_kyoto_ghg`. "
        "Currently kyoto_ghgs=('CO2', 'CH4', 'N2O')."
    )
    with pytest.raises(AssertionError, match=error_msg):
        calculate_kyoto_ghg(
            indf,
            kyoto_ghgs=("CO2", "CH4", "N2O"),
        )


def test_calculate_kyoto_ghg_one_missing_error(indf_basic):
    indf = indf_basic.loc[
        ~(
            (indf_basic.index.get_level_values("variable") == "N2O")
            & (indf_basic.index.get_level_values("ms") == "b")
        )
    ]

    missing_kyoto_ghgs_df = pd.DataFrame(
        ["N2O"],
        columns=["missing_kyoto_ghgs"],
        index=pd.MultiIndex.from_tuples(
            [("b",)],
            names=["ms"],
        ),
    )
    error_msg = re.escape(
        "For some groups, you are missing some Kyoto GHGs. "
        "Please either supply these gases for these groups "
        "or provide a different value for `kyoto_ghgs` to `calculate_kyoto_ghg`. "
        "Currently kyoto_ghgs=('CO2', 'CH4', 'N2O'). "
        "The groups and their missing Kyoto GHGs are:\n"
        f"{missing_kyoto_ghgs_df}"
    )
    with pytest.raises(AssertionError, match=error_msg):
        calculate_kyoto_ghg(
            indf,
            kyoto_ghgs=("CO2", "CH4", "N2O"),
        )


def test_calculate_kyoto_ghg_one_missing_error_multiple_other_groups():
    indf = pd.DataFrame(
        [
            [100, 110, 120],
            # [10, 11, 12],
            [1000.0, 2000.0, 3000.0],
            [200, 100, 300],
            [5, 6, 7],
            [1000.0, 500.0, 0.0],
            [200, 100, 300],
            [5, 6, 7],
            # [1000.0, 500.0, 0.0],
            # [200, 100, 300],
            [5, 6, 7],
            # [1000.0, 500.0, 0.0],
        ],
        columns=[2010, 2020, 2025],
        index=pd.MultiIndex.from_tuples(
            [
                ("a", "a", "CO2", "MtCO2 / yr"),
                # ("a", "a", "CH4", "MtCH4 / yr"),
                ("a", "a", "N2O", "ktN2O / yr"),
                ("a", "b", "CO2", "MtCO2 / yr"),
                ("a", "b", "CH4", "MtCH4 / yr"),
                ("a", "b", "N2O", "ktN2O / yr"),
                ("b", "a", "CO2", "MtCO2 / yr"),
                ("b", "a", "CH4", "MtCH4 / yr"),
                # ("b", "a", "N2O", "ktN2O / yr"),
                # ("b", "b", "CO2", "MtCO2 / yr"),
                ("b", "b", "CH4", "MtCH4 / yr"),
                # ("b", "b", "N2O", "ktN2O / yr"),
            ],
            names=["model", "scenario", "variable", "unit"],
        ),
    )

    missing_kyoto_ghgs_df = pd.DataFrame(
        ["CH4", "N2O", "CO2, N2O"],
        columns=["missing_kyoto_ghgs"],
        index=pd.MultiIndex.from_tuples(
            [("a", "a"), ("b", "a"), ("b", "b")],
            names=["model", "scenario"],
        ),
    )
    error_msg = re.escape(
        "For some groups, you are missing some Kyoto GHGs. "
        "Please either supply these gases for these groups "
        "or provide a different value for `kyoto_ghgs` to `calculate_kyoto_ghg`. "
        "Currently kyoto_ghgs=('CO2', 'CH4', 'N2O'). "
        "The groups and their missing Kyoto GHGs are:\n"
        f"{missing_kyoto_ghgs_df}"
    )
    with pytest.raises(AssertionError, match=error_msg):
        calculate_kyoto_ghg(
            indf,
            kyoto_ghgs=("CO2", "CH4", "N2O"),
        )


CMIP7_SCENARIOMIP_OUT_DIR = (
    Path(__file__).parents[1]
    / "regression/cmip7-scenariomip"
    / "cmip7-scenariomip-output"
)


def test_calculate_kyoto_ghg_kyoto_ghgs_and_naming_convention_not_supplied_error():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "If `kyoto_ghgs` is `None`, `indf_naming_convention` must be supplied"
        ),
    ):
        calculate_kyoto_ghg("not used")


@pytest.fixture(scope="module")
def calculate_kyoto_ghg_naming_conventions_reusable_inputs():
    indf = pandas_openscm.io.load_timeseries_csv(
        CMIP7_SCENARIOMIP_OUT_DIR
        / "COFFEE 1.6_SSP2 - Medium-Low Emissions_complete.csv",
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )

    exp = calculate_kyoto_ghg(
        indf,
        indf_naming_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    )

    return indf, exp


@pytest.mark.parametrize(
    "indf_naming_convention",
    (nc for nc in SupportedNamingConventions),
)
def test_calculate_kyoto_ghg_naming_conventions(
    indf_naming_convention, calculate_kyoto_ghg_naming_conventions_reusable_inputs
):
    indf_original_naming_convention, exp = (
        calculate_kyoto_ghg_naming_conventions_reusable_inputs
    )

    indf = rename_variables(
        indf_original_naming_convention,
        from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        to_convention=indf_naming_convention,
    )

    res = calculate_kyoto_ghg(
        indf,
        indf_naming_convention=indf_naming_convention,
    )

    pandas_openscm.testing.assert_frame_alike(res, exp)


@pytest.mark.parametrize("gwp", ("AR6GWP100", "AR5GWP100", "AR6GWP20"))
def test_calculate_kyoto_ghg_gwp(indf_basic, gwp):
    res = calculate_kyoto_ghg(indf_basic, kyoto_ghgs=("CO2", "CH4", "N2O"), gwp=gwp)

    ur = openscm_units.unit_registry
    with ur.context(gwp):
        exp = set_index_levels_func(
            groupby_except(
                pandas_openscm.unit_conversion.convert_unit(
                    indf_basic,
                    desired_units="MtCO2 / yr",
                    ur=ur,
                ),
                "variable",
            ).sum(),
            {"variable": "Kyoto GHG"},
        )
        if exp.columns.dtype != indf_basic.columns.dtype:
            # TODO: remove when this is no longer used
            # (likely when we upgrade supported pandas versions and python versions)
            exp.columns = exp.columns.astype(indf_basic.columns.dtype)

    pandas_openscm.testing.assert_frame_alike(res, exp)


@pytest.mark.parametrize(
    (
        "out_variable",
        "out_variable_exp",
        "out_unit",
        "out_unit_exp",
        "variable_level",
        "variable_level_exp",
        "unit_level",
        "unit_level_exp",
    ),
    (
        pytest.param(
            None,
            "Kyoto GHG",
            None,
            "MtCO2 / yr",
            None,
            "variable",
            None,
            "unit",
            id="defaults",
        ),
        pytest.param(
            "KGS",
            "KGS",
            "Gt C / yr",
            "Gt C / yr",
            "variables",
            "variables",
            "units",
            "units",
            id="all-user-specified",
        ),
    ),
)
def test_calculate_kyoto_ghg_metadata_arguments(  # noqa: PLR0913
    indf_basic,
    out_variable,
    out_variable_exp,
    out_unit,
    out_unit_exp,
    variable_level,
    variable_level_exp,
    unit_level,
    unit_level_exp,
):
    indf = indf_basic.copy()

    ur = openscm_units.unit_registry
    with ur.context("AR6GWP100"):
        exp = set_index_levels_func(
            groupby_except(
                pandas_openscm.unit_conversion.convert_unit(
                    indf_basic,
                    desired_units=out_unit_exp,
                    ur=ur,
                ),
                "variable",
            ).sum(),
            {variable_level_exp: out_variable_exp},
        )
        if exp.columns.dtype != indf_basic.columns.dtype:
            # TODO: remove when this is no longer used
            # (likely when we upgrade supported pandas versions and python versions)
            exp.columns = exp.columns.astype(indf_basic.columns.dtype)

    exp.index = exp.index.rename({"unit": unit_level_exp})

    call_kwargs = {}

    if out_variable is not None:
        call_kwargs["out_variable"] = out_variable

    if out_unit is not None:
        call_kwargs["out_unit"] = out_unit

    if variable_level is not None:
        call_kwargs["variable_level"] = variable_level
        indf.index = indf.index.rename({"variable": variable_level})

    if unit_level is not None:
        call_kwargs["unit_level"] = unit_level
        indf.index = indf.index.rename({"unit": unit_level})

    res = calculate_kyoto_ghg(
        indf,
        kyoto_ghgs=("CO2", "CH4", "N2O"),
        **call_kwargs,
    )

    pandas_openscm.testing.assert_frame_alike(res, exp)


def test_calculate_kyoto_ghg_ur_injection(indf_basic):
    ur = copy.deepcopy(openscm_units.unit_registry)

    gwpzn_context = pint.Context("GWPZN")
    gwpzn_context = ur._add_transformations_to_context(
        # gwpzn_context, "[methane]", ur.CH4, "[carbon]", ur.CO2, 30.0 * 12.0 / 44.0
        gwpzn_context,
        "[methane]",
        ur.CH4,
        "[carbon]",
        ur.CO2,
        30.0,
    )
    gwpzn_context = ur._add_transformations_to_context(
        gwpzn_context,
        "[nitrous_oxide]",
        ur.N2O,
        "[carbon]",
        ur.CO2,
        280.0,
    )
    ur.add_context(gwpzn_context)

    res = calculate_kyoto_ghg(
        indf_basic,
        kyoto_ghgs=("CO2", "CH4", "N2O"),
        gwp="GWPZN",
        ur=ur,
    )

    exp = pd.DataFrame(
        [
            [
                100 + 10 * 30.0 + 1.0 * 280.0,
                110 + 11 * 30.0 + 2.0 * 280.0,
                120 + 12 * 30.0 + 3.0 * 280.0,
            ],
            [
                200 + 5 * 30.0 + 1.0 * 280.0,
                100 + 6 * 30.0 + 0.5 * 280.0,
                300 + 7 * 30.0 + 0.0 * 280.0,
            ],
        ],
        columns=[2010, 2020, 2025],
        index=pd.MultiIndex.from_tuples(
            [
                ("a", "Kyoto GHG", "MtCO2 / yr"),
                ("b", "Kyoto GHG", "MtCO2 / yr"),
            ],
            names=["ms", "variable", "unit"],
        ),
    )

    pandas_openscm.testing.assert_frame_alike(res, exp)
