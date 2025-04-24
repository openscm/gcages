"""
Integration tests of our pre-processing for CMIP7 ScenarioMIP
"""

import itertools
from contextlib import nullcontext as does_not_raise
from functools import partial

import numpy as np
import pandas as pd
import pytest
from pandas_openscm.grouping import groupby_except
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.indexing import multi_index_lookup

from gcages.cmip7_scenariomip import (
    CMIP7ScenarioMIPPreProcessor,
    InternalConsistencyError,
)
from gcages.cmip7_scenariomip.pre_processing.gridding_emissions_to_global_workflow_emissions import (
    convert_gridding_emissions_to_global_workflow_emissions,
)
from gcages.completeness import assert_all_groups_are_complete
from gcages.index_manipulation import (
    combine_sectors,
    combine_species,
    set_new_single_value_levels,
    split_sectors,
)
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import (
    assert_frame_equal,
    get_cmip7_scenariomip_like_input,
)
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

pix = pytest.importorskip("pandas_indexing")


def test_aviation_reaggregation(example_input_output):
    """
    Test the re-aggregation of the aviation data

    Domestic and international aviation should be added to make |Aircraft.
    Transportation should have domestic aviation
    subtracted to make |Transportation Sector.
    """
    # Not interested in global level for this
    res = example_input_output.output.gridding_workflow_emissions

    # The original Transportation should be dropped out
    assert (
        not res.pix.unique("variable")
        .str.endswith("Energy|Demand|Transportation")
        .any()
    ), res.pix.unique("variable")

    exp_aircraft = combine_sectors(
        groupby_except(
            split_sectors(example_input_output.input).loc[
                # Only expect results reported at the World level
                pix.ismatch(sectors="**Aviation", region="World")
            ],
            "sectors",
        )
        .sum()
        .pix.assign(sectors="Aircraft")
    )

    assert_frame_equal(
        res.loc[pix.ismatch(variable="**Aircraft")],
        exp_aircraft,
    )

    tmp = (
        split_sectors(example_input_output.input.loc[~pix.isin(region="World")])
        .stack()
        .unstack("sectors")
    )
    tmp["Transportation Sector"] = (
        tmp["Energy|Demand|Transportation"]
        - tmp["Energy|Demand|Transportation|Domestic Aviation"]
    )
    exp_transport = combine_sectors(
        tmp[["Transportation Sector"]].stack().unstack("year")
    )

    assert_frame_equal(
        res.loc[pix.ismatch(variable="**Transportation Sector")],
        exp_transport,
    )


def test_industrial_sector_aggregation(example_input_output):
    # Not interested in global level for this
    res = example_input_output.output.gridding_workflow_emissions

    exp_sector = "Industrial Sector"
    exp_contributing_sectors = [
        "Energy|Demand|Industry",
        "Energy|Demand|Other Sector",
        "Industrial Processes",
        "Other",
    ]

    exp = combine_sectors(
        groupby_except(
            split_sectors(example_input_output.input).loc[
                # Only expect results reported at the regional level
                pix.isin(sectors=exp_contributing_sectors) & ~pix.isin(region="World")
            ],
            "sectors",
        )
        .sum()
        .pix.assign(sectors=exp_sector)
    )

    assert_frame_equal(res.loc[pix.ismatch(variable=f"**{exp_sector}")], exp)


def test_agricultural_sector_aggregation(example_input_output):
    # Not interested in global level for this
    res = example_input_output.output.gridding_workflow_emissions

    exp_sector = "Agriculture"
    exp_contributing_sectors = [
        "AFOLU|Agriculture",
        "AFOLU|Land|Land Use and Land-Use Change",
        "AFOLU|Land|Harvested Wood Products",
        "AFOLU|Land|Other",
        "AFOLU|Land|Wetlands",
    ]

    exp = combine_sectors(
        groupby_except(
            split_sectors(example_input_output.input).loc[
                # Only expect results reported at the regional level
                pix.isin(sectors=exp_contributing_sectors) & ~pix.isin(region="World")
            ],
            "sectors",
        )
        .sum()
        .pix.assign(sectors=exp_sector)
    )

    assert_frame_equal(res.loc[pix.ismatch(variable=f"**{exp_sector}")], exp)


def test_output_index(example_input_output):
    exp_output_species = [
        "CO2",
        "CH4",
        "N2O",
        "BC",
        "CO",
        "NH3",
        "OC",
        "NOx",
        "Sulfur",
        "VOC",
    ]

    exp_output_sectors_world = [
        "Aircraft",
        "International Shipping",
    ]

    exp_output_sectors_model_region = [
        "Energy Sector",
        "Industrial Sector",
        "Residential Commercial Other",
        "Solvents Production and Application",
        "Transportation Sector",
        "Waste",
        "Agriculture",
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Peat Burning",
    ]

    exp_index = pd.MultiIndex.from_tuples(
        [
            *[
                (f"Emissions|{species}|{sector}", "World")
                for species, sector in itertools.product(
                    exp_output_species, exp_output_sectors_world
                )
            ],
            *[
                (f"Emissions|{species}|{sector}", region)
                for species, sector, region in itertools.product(
                    exp_output_species,
                    exp_output_sectors_model_region,
                    example_input_output.model_regions,
                )
            ],
        ],
        names=["variable", "region"],
    )

    assert_all_groups_are_complete(
        example_input_output.output.gridding_workflow_emissions, exp_index
    )


def test_output_consistency_with_input_for_non_region_sector(example_input_output):
    """
    Test consistency between the output that is not at the region-sector level
    and the input emissions for species that aren't used in gridding
    """
    gridding_species = split_sectors(
        example_input_output.output.gridding_workflow_emissions
    ).pix.unique("species")

    not_from_region_sector = [
        v
        for v in example_input_output.output.global_workflow_emissions_raw_names.pix.unique(  # noqa: E501
            "variable"
        )
        if not any(species in v for species in gridding_species)
    ]

    not_from_region_sector_res = (
        example_input_output.output.global_workflow_emissions_raw_names.loc[
            pix.isin(variable=not_from_region_sector)
        ]
    )

    not_from_region_sector_compare = strip_pint_incompatible_characters_from_units(
        example_input_output.input.loc[pix.isin(variable=not_from_region_sector)]
    )

    assert_frame_equal(not_from_region_sector_res, not_from_region_sector_compare)


def test_output_internal_consistency(example_input_output):
    """
    Test consistency between the output that is not at the region-sector level
    and the output that is at the world level
    """
    global_workflow_emissions_derived = (
        convert_gridding_emissions_to_global_workflow_emissions(
            example_input_output.output.gridding_workflow_emissions,
            global_workflow_co2_fossil_sector="Energy and Industrial Processes",
            global_workflow_co2_biosphere_sector="AFOLU",
        )
    )

    exp_compare = multi_index_lookup(
        example_input_output.output.global_workflow_emissions_raw_names,
        global_workflow_emissions_derived.index,
    )
    assert_frame_equal(exp_compare, global_workflow_emissions_derived)


def test_output_internal_consistency_global_workflow_emissions(example_input_output):
    assert_frame_equal(
        update_index_levels_func(
            example_input_output.output.global_workflow_emissions_raw_names,
            dict(
                variable=partial(
                    convert_variable_name,
                    from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            ),
        ),
        example_input_output.output.global_workflow_emissions,
    )


def test_output_vs_start_region_sector_consistency(example_input_output):
    # We have renamed all the sectors
    # and moved domestic aviation to global only in the output,
    # so we can't check the regional sum for each sector.
    # Hence we jump straight to checking the regional sum of our sectoral sum
    # i.e. the total.
    gridded_emisssions_sectoral_regional_sum = set_new_single_value_levels(
        combine_species(
            groupby_except(
                split_sectors(
                    example_input_output.output.gridding_workflow_emissions,
                    bottom_level="sectors",
                ),
                ["region", "sectors"],
            ).sum()
        ),
        {"region": "World"},
    )

    in_emissions_totals_to_compare_to = multi_index_lookup(
        example_input_output.input,
        gridded_emisssions_sectoral_regional_sum.index,
    )
    assert_frame_equal(
        in_emissions_totals_to_compare_to,
        gridded_emisssions_sectoral_regional_sum,
    )


REQUIRED_SECTORS_REGIONAL = (
    "Energy|Supply",
    "Energy|Demand|Industry",
    "Energy|Demand|Residential and Commercial and AFOFI",
    "Energy|Demand|Transportation",
    "Energy|Demand|Transportation|Domestic Aviation",
    "Industrial Processes",
    "Product Use",
    "Waste",
    "AFOLU|Agriculture",
    "AFOLU|Agricultural Waste Burning",
    "AFOLU|Land|Fires|Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning",
)

REQUIRED_SECTORS_WORLD = (
    "Energy|Demand|Bunkers|International Aviation",
    "Energy|Demand|Bunkers|International Shipping",
)

OPTIONAL_SECTORS = (
    "Energy|Demand|Other Sector",
    "Other",
    "Other Capture and Removal",
    "AFOLU|Land|Land Use and Land-Use Change",
    "AFOLU|Land|Harvested Wood Products",
    "AFOLU|Land|Other",
    "AFOLU|Land|Wetlands",
    "AFOLU|Land|Fires|Peat Burning",
)


@pytest.mark.parametrize(
    "sector_to_delete, exp",
    (
        *(
            pytest.param(v, pytest.raises(ValueError), id=f"{v}_not-complete-error")
            for v in [*REQUIRED_SECTORS_REGIONAL, *REQUIRED_SECTORS_WORLD]
        ),
        *(
            pytest.param(
                v,
                pytest.raises(InternalConsistencyError),
                id=f"{v}_internal-consistency-error",
            )
            # These don't cause completeness issues
            # (because they're not required),
            # but they do break the internal consistency of the data.
            for v in OPTIONAL_SECTORS
        ),
        *(
            pytest.param(v, does_not_raise(), id=f"{v}_can-be-missing")
            for v in [
                # Sectors what we don't consider at all
                "Energy|Demand",
                "Energy",
            ]
        ),
    ),
)
def test_input_missing_variable(sector_to_delete, exp, complete_input):
    inp = complete_input

    inp = inp.loc[~pix.ismatch(variable=f"**CO2|{sector_to_delete}")]

    with exp:
        # Checks on by default
        CMIP7ScenarioMIPPreProcessor(n_processes=None, progress=False)(inp)


@pytest.mark.parametrize(
    "sector_to_modify, exp",
    (
        *(
            pytest.param(
                v,
                pytest.raises(InternalConsistencyError),
                id=f"{v}_internal-consistency-error",
            )
            for v in [
                *REQUIRED_SECTORS_REGIONAL,
                *REQUIRED_SECTORS_WORLD,
                *OPTIONAL_SECTORS,
            ]
            # domestic aviation isn't used in the internal consistency checks,
            # even though it is required to be reported
            if "Domestic Aviation" not in v
        ),
        *(
            pytest.param(v, does_not_raise(), id=f"{v}_can-be-missing")
            for v in [
                # Not used for internal consistency checking
                "Energy|Demand|Transportation|Domestic Aviation",
                # Sectors what we don't consider at all
                "Energy|Demand",
                "Energy",
                # TODO: use this properly
                "Other Capture and Removal",
            ]
        ),
    ),
)
def test_input_not_internally_consistent_error_incorrect_sum(
    sector_to_modify, exp, complete_input
):
    inp = complete_input

    # Modify a variable without altering the rest of the tree to match
    inp.loc[pix.ismatch(variable=f"**NOx|{sector_to_modify}")] *= 1.1

    with exp:
        # Checks on by default
        CMIP7ScenarioMIPPreProcessor(
            n_processes=None,
            progress=False,
        )(inp)


def test_multiple_scenarios_different_time_axes():
    scenario_1 = get_cmip7_scenariomip_like_input(
        timesteps=np.arange(2010, 2100 + 1, 10), model="model_1", scenario="scenario_1"
    )
    scenario_2 = get_cmip7_scenariomip_like_input(
        # five year steps, not ten, and starting in 2015
        timesteps=np.arange(2015, 2100 + 1, 5),
        model="model_2",
        scenario="scenario_a",
        regions=("India", "Brazil", "North America"),
    )

    pre_processor = CMIP7ScenarioMIPPreProcessor(
        progress=False,
        n_processes=None,  # process serially
    )

    res_1 = pre_processor(scenario_1)
    res_2 = pre_processor(scenario_2)

    scenarios_combined = pix.concat([scenario_1, scenario_2]).sort_index(axis="columns")
    res_combined = pre_processor(scenarios_combined)

    for res_individual in [res_1, res_2]:
        for attr in [
            "gridding_workflow_emissions",
            "global_workflow_emissions",
            "global_workflow_emissions_raw_names",
        ]:
            res_individual_df = getattr(res_individual, attr)

            model_l = pix.uniquelevel(res_individual_df, "model")
            if len(model_l) != 1:
                raise AssertionError
            model = model_l[0]
            scenario_l = pix.uniquelevel(res_individual_df, "scenario")
            if len(scenario_l) != 1:
                raise AssertionError
            scenario = scenario_l[0]

            res_combined_df = getattr(res_combined, attr)
            res_combined_df_ms = res_combined_df.loc[
                pix.isin(model=model, scenario=scenario)
            ]
            res_combined_df_ms_nan_times_dropped = res_combined_df_ms.dropna(
                how="all", axis="columns"
            )

            assert_frame_equal(res_individual_df, res_combined_df_ms_nan_times_dropped)
