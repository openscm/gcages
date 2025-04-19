"""
Integration tests of our pre-processing for CMIP7 ScenarioMIP
"""

import itertools
import re
from contextlib import nullcontext as does_not_raise
from functools import partial

import pytest
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
from gcages.cmip7_scenariomip.pre_processing import (
    InternalConsistencyError,
    get_gridded_emissions_sectoral_regional_sum,
)
from gcages.completeness import NotCompleteError
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
def example_complete_input():
    return get_cmip7_scenariomip_like_input()


@pytest.fixture(scope="session")
def processed_output(example_complete_input):
    pre_processor = CMIP7ScenarioMIPPreProcessor(
        n_processes=None,  # run serially
    )

    processed = pre_processor(example_complete_input)

    return processed


def test_transport_shuffling(example_complete_input, processed_output):
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

    example_complete_input_sectors = unstack_sector(example_complete_input)

    tmp = example_complete_input_sectors.copy()
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

    tmp = example_complete_input_sectors.copy()
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


def test_industrial_sector_aggregation(example_complete_input, processed_output):
    # Not interested in global level for this
    df_to_check = processed_output.gridding_workflow_emissions

    exp_sector = "Industrial Sector"
    exp_contributing_sectors = [
        "Energy|Demand|Industry",
        "Energy|Demand|Other Sector",
        "Industrial Processes",
        "Other",
    ]

    example_complete_input_sectors = unstack_sector(example_complete_input)
    tmp = example_complete_input_sectors.copy()

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


def test_agricultural_sector_aggregation(example_complete_input, processed_output):
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

    example_complete_input_sectors = unstack_sector(example_complete_input)
    tmp = example_complete_input_sectors.copy()
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


def test_output_sectors(example_complete_input, processed_output):
    example_complete_input_sectors = unstack_sector(example_complete_input)

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
            example_complete_input_sectors.pix.unique("table"),
            example_complete_input_sectors.loc[~pix.isin(region="World")].pix.unique(
                "species"
            ),
            exp_output_sectors,
        )
    ]

    assert set(exp_output_variables) == set(
        processed_output.gridding_workflow_emissions.pix.unique("variable")
    )


def test_output_consistency_with_input_for_non_region_sector(
    example_complete_input, processed_output
):
    """
    Test consistency between
    `processed_output.global_workflow_emissions`
    and the input emission
    """
    region_sector_split_species = unstack_sector(
        processed_output.gridding_workflow_emissions
    ).pix.unique("species")

    not_from_region_sector = [
        v
        for v in processed_output.global_workflow_emissions_raw_names.pix.unique(
            "variable"
        )
        if not any(species in v for species in region_sector_split_species)
    ]

    not_from_region_sector_res = (
        processed_output.global_workflow_emissions_raw_names.loc[
            pix.isin(variable=not_from_region_sector)
        ]
    )

    not_from_region_sector_compare = strip_pint_incompatible_characters_from_units(
        example_complete_input.loc[pix.isin(variable=not_from_region_sector)]
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
        processed_output.gridding_workflow_emissions.pix.unique("variable").map(
            lambda x: x.count("|")
        )
        == 2
    ).all()
    sector_cols = unstack_sector(processed_output.gridding_workflow_emissions)

    non_sector_region_group_levels = sector_cols.index.names.difference(
        ["sector", "region"]
    )
    region_sector_totals = (
        sector_cols.sum(axis="columns")
        .groupby(non_sector_region_group_levels)
        .sum()
        .pix.assign(region="World")
    )

    region_sector_compare_non_co2 = (
        region_sector_totals.unstack("year")
        .loc[~pix.isin(species="CO2")]
        .pix.format(variable="{table}|{species}", drop=True)
    )

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
    biosphere_sectors = [
        "Agriculture",
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Peat Burning",
    ]
    region_sector_compare_co2_fossil = stack_sector_and_return_to_variable(
        sector_cols.loc[pix.isin(species="CO2"), fossil_sectors]
        .groupby(non_sector_region_group_levels)
        .sum()
        .pix.assign(sectors="Energy and Industrial Processes", region="World")
        .sum(axis="columns")
        .unstack("sectors")
    )
    region_sector_compare_co2_afolu = stack_sector_and_return_to_variable(
        sector_cols.loc[pix.isin(species="CO2"), biosphere_sectors]
        .groupby(non_sector_region_group_levels)
        .sum()
        .pix.assign(sectors="AFOLU", region="World")
        .sum(axis="columns")
        .unstack("sectors")
    )

    region_sector_compare = pix.concat(
        [
            region_sector_compare_co2_fossil,
            region_sector_compare_co2_afolu,
            region_sector_compare_non_co2,
        ]
    )
    region_sector_split_species = sector_cols.pix.unique("species")
    from_region_sector = [
        v
        for v in processed_output.global_workflow_emissions_raw_names.pix.unique(
            "variable"
        )
        if any(species in v for species in region_sector_split_species)
    ]

    from_region_sector_locator = pix.isin(variable=from_region_sector)
    assert_frame_equal(
        processed_output.global_workflow_emissions_raw_names.loc[
            from_region_sector_locator
        ],
        region_sector_compare,
    )


def test_output_internal_consistency_global_workflow_emissions(processed_output):
    assert_frame_equal(
        update_index_levels_func(
            processed_output.global_workflow_emissions_raw_names,
            dict(
                variable=partial(
                    convert_variable_name,
                    from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            ),
        ),
        processed_output.global_workflow_emissions,
    )


def test_output_vs_start_region_sector_consistency(
    example_complete_input, processed_output
):
    # We have renamed all the sectors
    # and moved domestic aviation to global only in the output,
    # so we can't check the regional sum for each sector.
    # Hence we jump straight to checking the regional sum of our sectoral sum
    # i.e. the total.
    res_sectoral_regional_sum = get_gridded_emissions_sectoral_regional_sum(
        processed_output.gridding_workflow_emissions,
        time_name="year",
        region_level="region",
        world_region="World",
    )

    exp_sectoral_regional_sum = example_complete_input.loc[
        pix.ismatch(
            variable=res_sectoral_regional_sum.pix.unique("variable"),
            region=res_sectoral_regional_sum.pix.unique("region"),
        )
    ]

    assert_frame_equal(res_sectoral_regional_sum, exp_sectoral_regional_sum)


REQUIRED_SECTORS_REGIONAL = (
    "Energy|Supply",
    "Energy|Demand|Industry",
    "Energy|Demand|Other Sector",
    "Energy|Demand|Residential and Commercial and AFOFI",
    "Energy|Demand|Transportation",
    "Energy|Demand|Transportation|Domestic Aviation",
    "Industrial Processes",
    "Other",
    "Product Use",
    "Waste",
    "AFOLU|Agriculture",
    "AFOLU|Agricultural Waste Burning",
    "AFOLU|Land|Fires|Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning",
    "AFOLU|Land|Fires|Peat Burning",
)

REQUIRED_SECTORS_WORLD = (
    "Energy|Demand|Bunkers|International Aviation",
    "Energy|Demand|Bunkers|International Shipping",
)

OPTIONAL_SECTORS = (
    "AFOLU|Land|Land Use and Land-Use Change",
    "AFOLU|Land|Harvested Wood Products",
    "AFOLU|Land|Other",
    "AFOLU|Land|Wetlands",
)


@pytest.mark.parametrize(
    "sector_to_delete, exp",
    (
        *(
            pytest.param(
                v, pytest.raises(NotCompleteError), id=f"{v}_not-complete-error"
            )
            for v in [*REQUIRED_SECTORS_REGIONAL, *REQUIRED_SECTORS_WORLD]
        ),
        *(
            pytest.param(
                v,
                # pytest.raises(AssertionError, match=re.escape("junk")),
                pytest.raises(KeyError, match=re.escape("junk")),
                id=f"{v}_assertion-error",
            )
            # These don't cause completeness issues,
            # but they do break the internal consistency of the data.
            for v in OPTIONAL_SECTORS
        ),
        *(
            pytest.param(v, does_not_raise(), id=f"{v}_can-be-missing")
            for v in [
                # Sectors what we don't consider at all
                "Energy|Demand",
                "Energy",
                # # TODO: check whether we should be using this
                # # (I guess we'll find out once we start using IAM data)
                "Other Capture and Removal",
            ]
        ),
    ),
)
def test_input_missing_variable(sector_to_delete, exp, example_complete_input):
    inp = example_complete_input.copy()

    inp = inp.loc[~pix.ismatch(variable=f"**CO2|{sector_to_delete}")]

    with exp:
        # Checks on by default
        CMIP7ScenarioMIPPreProcessor()(inp)


def test_input_not_internally_consistent_error_incorrect_sum(example_complete_input):
    inp = example_complete_input.copy()

    # Modify a variable without altering the rest of the tree
    # (note that we only pick up variables we use, so e.g.
    # deleting "**CO2|Energy" does nothing)
    # TODO: parameterise so this is more obvious
    # Cases:
    # - variable we use at sectoral level: exp error
    # - variable we use only at global level: exp no error (mismatch doesn't matter for us)
    # - variable we don't use: exp no error (doesn't matter for us)
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
        CMIP7ScenarioMIPPreProcessor()(inp)


# Tests to write:
# - domestic aviation reported only globally blows up as expected
# - two scenarios that report on different time grids also works
#   (should be same result as handling them separately)
