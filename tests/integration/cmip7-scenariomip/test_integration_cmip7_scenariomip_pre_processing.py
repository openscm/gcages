"""
Integration tests of our pre-processing for CMIP7 ScenarioMIP
"""

import itertools
import re
from functools import partial

import pytest
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor

# from gcages.cmip7_scenariomip.pre_processing import InternalConsistencyError
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import (
    assert_frame_equal,
    get_cmip7_scenariomip_like_input,
    stack_sector_and_return_to_variable,
    unstack_sector,
)
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

pix = pytest.importorskip("pandas_indexing")


@pytest.fixture(scope="session")
def example_raw_input():
    return get_cmip7_scenariomip_like_input()


@pytest.fixture(scope="session")
def processed_output(example_raw_input):
    pre_processor = CMIP7ScenarioMIPPreProcessor(
        n_processes=None,  # run serially
    )

    processed = pre_processor(example_raw_input)

    return processed


def test_transport_shuffling(example_raw_input, processed_output):
    """
    Test the moving of the transport data

    Domestic and international aviation should be added to make |Aircraft.
    Transportation should have domestic aviation
    subtracted to make |Transportation Sector.
    """
    # Not interested in global level for this
    df_to_check = processed_output.gridding_workflow_emissions

    # The original Transportation should be dropped out
    assert (
        not df_to_check.pix.unique("variable")
        .str.endswith("Energy|Demand|Transportation")
        .any()
    ), df_to_check.pix.unique("variable")

    example_raw_input_sectors = unstack_sector(example_raw_input)

    tmp = example_raw_input_sectors.copy()
    tmp["Aircraft"] = tmp[
        [
            "Energy|Demand|Transportation|Domestic Aviation",
            "Energy|Demand|Bunkers|International Aviation",
        ]
    ].sum(axis="columns", min_count=2)
    exp_aircraft = stack_sector_and_return_to_variable(
        # Only expect results reported at the World level
        tmp[["Aircraft"]].dropna().loc[pix.ismatch(region="World")]
    )

    tmp = example_raw_input_sectors.copy()
    tmp["Transportation Sector"] = (
        tmp["Energy|Demand|Transportation"]
        - tmp["Energy|Demand|Transportation|Domestic Aviation"]
    )
    exp_transportation_sector = stack_sector_and_return_to_variable(
        # Only expect results reported at the regional level
        tmp[["Transportation Sector"]].dropna().loc[~pix.ismatch(region="World")]
    )

    assert_frame_equal(
        df_to_check.loc[pix.ismatch(variable="**Aircraft")],
        exp_aircraft,
    )

    assert_frame_equal(
        df_to_check.loc[pix.ismatch(variable="**Transportation Sector")],
        exp_transportation_sector,
    )


def test_industrial_sector_aggregation(example_raw_input, processed_output):
    # Not interested in global level for this
    df_to_check = processed_output.gridding_workflow_emissions

    exp_sector = "Industrial Sector"
    exp_contributing_sectors = [
        "Energy|Demand|Industry",
        "Energy|Demand|Other Sector",
        "Industrial Processes",
        "Other",
    ]

    example_raw_input_sectors = unstack_sector(example_raw_input)
    tmp = example_raw_input_sectors.copy()
    tmp[exp_sector] = tmp[exp_contributing_sectors].sum(
        axis="columns", min_count=len(exp_contributing_sectors)
    )
    exp_sector_df = stack_sector_and_return_to_variable(
        # Only expect results reported at the regional level
        tmp[[exp_sector]].dropna().loc[~pix.ismatch(region="World")]
    )

    assert_frame_equal(
        df_to_check.loc[pix.ismatch(variable=f"**{exp_sector}")],
        exp_sector_df,
    )


def test_agricultural_sector_aggregation(example_raw_input, processed_output):
    # Not interested in global level for this
    df_to_check = processed_output.gridding_workflow_emissions

    exp_sector = "Agriculture"
    exp_contributing_sectors = [
        "AFOLU|Agriculture",
        "AFOLU|Land|Land Use and Land-Use Change",
        "AFOLU|Land|Harvested Wood Products",
        "AFOLU|Land|Other",
        "AFOLU|Land|Wetlands",
    ]

    example_raw_input_sectors = unstack_sector(example_raw_input)
    tmp = example_raw_input_sectors.copy()
    tmp[exp_sector] = tmp[exp_contributing_sectors].sum(
        axis="columns", min_count=len(exp_contributing_sectors)
    )
    exp_sector_df = stack_sector_and_return_to_variable(
        # Only expect results reported at the regional level
        tmp[[exp_sector]].dropna().loc[~pix.ismatch(region="World")]
    )

    assert_frame_equal(
        df_to_check.loc[pix.ismatch(variable=f"**{exp_sector}")],
        exp_sector_df,
    )


def test_output_sectors(example_raw_input, processed_output):
    example_raw_input_sectors = unstack_sector(example_raw_input)

    exp_output_sectors = [
        # Fossil
        "Energy Sector",
        "Industrial Sector",
        "Residential Commercial Other",
        "Solvents Production and Application",
        "Transportation Sector",
        "Waste",
        # Bunkers
        "Aircraft",
        "International Shipping",
        # AFOLU
        "Agriculture",
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Peat Burning",
    ]

    exp_output_variables = [
        f"{table}|{gas}|{sector}"
        for table, gas, sector in itertools.product(
            example_raw_input_sectors.pix.unique("table"),
            example_raw_input_sectors.loc[~pix.isin(region="World")].pix.unique(
                "species"
            ),
            exp_output_sectors,
        )
    ]

    assert set(exp_output_variables) == set(
        processed_output.gridding_workflow_emissions.pix.unique("variable")
    )


def test_output_consistency_with_input_for_non_region_sector(
    example_raw_input, processed_output
):
    """
    Test consistency between
    `processed_output.global_workflow_emissions`
    and the input emission
    """
    region_sector_split_species = split_variable(
        processed_output.region_sector_workflow_emissions
    ).pix.unique("gas")

    not_from_region_sector = [
        v
        for v in processed_output.global_workflow_emissions.pix.unique("variable")
        if not any(species in v for species in region_sector_split_species)
    ]

    not_from_region_sector_res = processed_output.global_workflow_emissions.loc[
        pix.isin(variable=not_from_region_sector)
    ]

    not_from_region_sector_compare = strip_pint_incompatible_characters_from_units(
        example_raw_input.loc[pix.isin(variable=not_from_region_sector)]
    )

    assert_frame_equal(not_from_region_sector_res, not_from_region_sector_compare)


def test_output_internal_consistency(processed_output):
    """
    Test consistency between
    `processed_output.global_workflow_emissions`
    and `processed_output.region_sector_workflow_emissions`
    """
    # Make sure that we can just aggregate semi-blindly
    assert (
        processed_output.region_sector_workflow_emissions.pix.unique("variable").map(
            lambda x: x.count("|")
        )
        == 2
    ).all()
    region_sector_split = split_variable(
        processed_output.region_sector_workflow_emissions
    )

    non_sector_region_group_levels = region_sector_split.index.names.difference(
        ["sector", "region"]
    )
    region_sector_totals = (
        region_sector_split.groupby(non_sector_region_group_levels)
        .sum()
        .pix.assign(region="World")
    )

    region_sector_compare_non_co2 = region_sector_totals.loc[
        ~pix.isin(gas="CO2")
    ].pix.format(variable="{table}|{gas}", drop=True)

    fossil_sectors = [
        "Energy Sector",
        "Industrial Sector",
        "Residential Commercial Other",
        "Solvents Production and Application",
        "Transportation Sector",
        "Waste",
        "Aircraft",
        "International Shipping",
    ]
    region_sector_compare_co2_fossil = combine_to_make_variable(
        region_sector_split.loc[pix.isin(gas="CO2", sector=fossil_sectors)]
        .groupby(non_sector_region_group_levels)
        .sum()
        .pix.assign(sector="Energy and Industrial Processes", region="World")
    )
    region_sector_compare_co2_afolu = combine_to_make_variable(
        region_sector_split.loc[pix.isin(gas="CO2") & ~pix.isin(sector=fossil_sectors)]
        .groupby(non_sector_region_group_levels)
        .sum()
        .pix.assign(sector="AFOLU", region="World")
    )

    region_sector_compare = pix.concat(
        [
            region_sector_compare_co2_fossil,
            region_sector_compare_co2_afolu,
            region_sector_compare_non_co2,
        ]
    )
    region_sector_split_species = region_sector_split.pix.unique("gas")
    from_region_sector = [
        v
        for v in processed_output.global_workflow_emissions.pix.unique("variable")
        if any(species in v for species in region_sector_split_species)
    ]

    from_region_sector_locator = pix.isin(variable=from_region_sector)
    assert_frame_equal(
        processed_output.global_workflow_emissions.loc[from_region_sector_locator],
        region_sector_compare,
    )


def test_output_internal_consistency_gcages(processed_output):
    assert_frame_equal(
        update_index_levels_func(
            processed_output.global_workflow_emissions,
            dict(
                variable=partial(
                    convert_variable_name,
                    from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            ),
        ),
        processed_output.global_workflow_emissions_gcages,
    )


def test_output_vs_start_region_sector_consistency(example_raw_input, processed_output):
    # TODO: put this in `self.run_checks` too
    to_check = processed_output.region_sector_workflow_emissions
    to_check_split = split_variable(to_check)

    res_sectoral_sum = (
        to_check_split.groupby(to_check_split.index.names.difference(["sector"]))
        .sum()
        .pix.format(variable="{table}|{gas}", drop=True)
    )
    exp_sectoral_sum = example_raw_input.loc[
        pix.ismatch(
            variable=res_sectoral_sum.pix.unique("variable"),
            region=res_sectoral_sum.pix.unique("region"),
        )
    ]
    assert_frame_equal(res_sectoral_sum, exp_sectoral_sum)

    # We have renamed all the sectors in the output,
    # so we can't check the regional sum for each sector.
    # Hence we jump straight to checking the regional sum of our sectoral sum.
    res_sectoral_regional_sum = (
        # Implicitly also testing that World is not in the output
        # (users would have to aggregate that themselves)
        res_sectoral_sum.groupby(res_sectoral_sum.index.names.difference(["region"]))
        .sum()
        .pix.assign(region="World")
    )
    exp_sectoral_regional_sum = example_raw_input.loc[
        pix.ismatch(
            variable=res_sectoral_regional_sum.pix.unique("variable"),
            region=res_sectoral_regional_sum.pix.unique("region"),
        )
    ]
    assert_frame_equal(res_sectoral_regional_sum, exp_sectoral_regional_sum)


def test_input_not_internally_consistent_error_obvious(
    example_raw_input, default_data_structure_definition
):
    inp = example_raw_input.copy()

    # Just delete a high level variable
    inp = inp.loc[~pix.ismatch(variable="**CO2|Energy")]

    exp_components = sorted(
        default_data_structure_definition.variable["Emissions|CO2"].components
    )
    exp_components_included = [
        "Emissions|CO2|AFOLU",
        "Emissions|CO2|Industrial Processes",
        "Emissions|CO2|Other",
        "Emissions|CO2|Product Use",
        "Emissions|CO2|Waste",
    ]
    exp_components_missing = [
        "Emissions|CO2|Energy",
        "Emissions|CO2|Other Capture and Removal",
    ]

    exp_error_lines_except_df = f"""There are reporting issues in your data.
The issue occurs for the following variables:

1. Emissions|CO2
  - these are the expected component variables from the data structure definition: {exp_components}
  - if you have reported data that is not part of the expected component variables, this may be the issue
  - of the given components, these variables:
    - are included in the data: {exp_components_included}
    - are missing from the data: {exp_components_missing}

Here is a view of the difference between the variable values
and the sum of their expected components:"""  # noqa: E501
    with pytest.raises(
        InternalConsistencyError,
        match=(
            "".join(
                [
                    re.escape(exp_error_lines_except_df),
                    # Check the DataFrame is shown too
                    r"\s*year\s*2015.*",
                    r"\s*",
                    r"\s*".join(
                        [
                            "model",
                            "scenario",
                            "region",
                            "variable",
                            "unit",
                            "aggregation",
                        ]
                    ),
                ]
            )
        ),
    ):
        # Checks on by default
        CMIP7ScenarioMIPPreProcessor(
            data_structure_definition=default_data_structure_definition
        )(inp)


def test_input_not_internally_consistent_error_mismatch(
    example_raw_input, default_data_structure_definition
):
    inp = example_raw_input.copy()

    # Modify a variable without altering the rest of the tree
    inp.loc[pix.ismatch(variable="**CO2|Energy")] *= 3.0

    exp_components_included = [
        "Emissions|CO2|AFOLU",
        "Emissions|CO2|Energy",
        "Emissions|CO2|Industrial Processes",
        "Emissions|CO2|Other",
        "Emissions|CO2|Product Use",
        "Emissions|CO2|Waste",
    ]
    exp_components_missing = [
        "Emissions|CO2|Other Capture and Removal",
    ]
    exp_error_lines_to_check = f"""  - of the given components, these variables:
    - are included in the data: {exp_components_included}
    - are missing from the data: {exp_components_missing}"""

    with pytest.raises(
        InternalConsistencyError, match=re.escape(exp_error_lines_to_check)
    ):
        # Checks on by default
        CMIP7ScenarioMIPPreProcessor(
            data_structure_definition=default_data_structure_definition
        )(inp)


def test_input_not_internally_consistent_error_deep(
    example_raw_input, default_data_structure_definition
):
    # Will need to update common-definitions to make this work
    inp = example_raw_input.copy()
    # Add an extra variable
    to_add = inp.loc[
        pix.ismatch(variable="Emissions|CO2|Energy|Demand|Transportation|Rail")
    ].pix.assign(
        variable="Emissions|CO2|Energy|Demand|Transportation|Light-Duty Vehicle"
    )
    inp = pix.concat([inp, to_add])

    exp_error_lines_except_df = """There are reporting issues in your data.
The issue occurs for the following variables:

1. Emissions|CO2|Energy|Demand|Transportation
  - the variable has no components in the data structure definition, so we assume that we are insted checking over the variables in the hierarchy: ['Emissions|CO2|Energy|Demand|Transportation|Domestic Aviation', 'Emissions|CO2|Energy|Demand|Transportation|Rail', 'Emissions|CO2|Energy|Demand|Transportation|Light-Duty Vehicle']

Here is a view of the difference between the variable values
and the sum of their expected components:"""  # noqa: E501
    with pytest.raises(
        InternalConsistencyError,
        match=(
            "".join(
                [
                    re.escape(exp_error_lines_except_df),
                    # Check the DataFrame is shown too
                    r"\s*year\s*2015.*",
                    r"\s*",
                    r"\s*".join(
                        [
                            "model",
                            "scenario",
                            "region",
                            "variable",
                            "unit",
                            "aggregation",
                        ]
                    ),
                ]
            )
        ),
    ):
        # Checks on by default
        CMIP7ScenarioMIPPreProcessor(
            data_structure_definition=default_data_structure_definition
        )(inp)


@pytest.mark.parametrize(
    "to_remove",
    (
        "Energy|Supply",
        # "Industrial Sector",  # aggregated internally from the below
        "Energy|Demand|Industry",
        "Energy|Demand|Other Sector",
        "Industrial Processes",
        "Other",
        "Energy|Demand|Residential and Commercial and AFOFI",
        "Product Use",
        # "Transportation Sector",  # aggregated internally from the below
        "Energy|Demand|Transportation|Domestic Aviation",
        "Energy|Demand|Bunkers|International Aviation",
        "Energy|Demand|Transportation",
        "Waste",
        "Aircraft",
        "Energy|Demand|Bunkers|International Shipping",
        # "CEDS Agriculture",  # aggregated internally from the below
        "AFOLU|Agriculture",
        "AFOLU|Land|Land Use and Land-Use Change",
        "AFOLU|Land|Harvested Wood Products",
        "AFOLU|Land|Other",
        "AFOLU|Land|Wetlands",
        "AFOLU|Agricultural Waste Burning",
        "AFOLU|Land|Fires|Forest Burning",
        "AFOLU|Land|Fires|Grassland Burning",
        pytest.param(
            "AFOLU|Land|Fires|Peat Burning",
            marks=[pytest.mark.xfail(reason="Internal hack being applied")],
        ),
    ),
)
def test_input_missing_key_variable_error(example_raw_input, to_remove):
    pytest.xfail(reason="Need to think more carefully about this")
    # It's complicated because we should catch it before the sum,
    # because it's hard to clearly identify why the sum failed
    # (often you just get silent passes with a bunch of NaNs
    # because the sector is there for one variable but not another).
    # It's hard because it's also not clear what should be a hard fail
    # and what is allowed.
    inp = example_raw_input.copy()
    inp = inp.loc[~pix.ismatch(variable=f"**CH4|{to_remove}")]
    inp.pix.unique("variable")

    with pytest.raises(KeyError, match=re.escape("junk")):
        # Checks on by default
        CMIP7ScenarioMIPPreProcessor()(inp)


def test_input_not_internally_consistent_error_regional(example_raw_input):
    # It might be impossible to get regional reporting errors
    # because we can just throw away world data
    # (e.g. if there is nothing which is reported only at the World level,
    # which would be a change from CMIP6 where international shipping and aviation
    # was only reported at the global level).
    # We should be able to check this by checking that, if we sum up
    # regional and sectoral data, we get World totals for gases
    # (TODO: this test on actual data).
    assert False, "TODO: ask Jarmo whether this is even possible"


# Underlying logic:
# - we're doing region-sector harmonisation
# - hence we need regions and sectors lined up very specifically with CEDS
# - hence this kind of pre-processing
#   (if you want different pre-processing, use something more like AR6)
