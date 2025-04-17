"""
Integration tests of our pre-processing for CMIP7 ScenarioMIP
"""

import itertools
from functools import partial

import numpy as np
import pandas as pd
import pytest
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import assert_frame_equal
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

pix = pytest.importorskip("pandas_indexing")

RNG = np.random.default_rng()


def aggregate_up_sectors(indf, copy=False):
    res = indf
    if copy:
        res = res.copy()

    aggregations = set(["|".join(c.split("|")[:-1]) for c in res if "|" in c])
    # Have to aggregate lowest level sectors first
    aggregations_sorted = sorted(aggregations, key=lambda x: x.count("|"))[::-1]
    for aggregation in aggregations_sorted:
        if aggregation in res:
            msg = f"{aggregation} already in indf?!"
            raise KeyError(msg)

        contributing = []
        for c in res:
            if not c.startswith(f"{aggregation}|"):
                continue

            split = c.split(f"{aggregation}|")
            if "|" in split[-1]:
                # going too many levels deep
                continue

            contributing.append(c)

        res[aggregation] = res[contributing].sum(axis="columns")

    return res


def split_variable(df):
    res = df.pix.extract(variable="{table}|{gas}|{sector}")

    return res


def combine_to_make_variable(df):
    res = df.pix.format(variable="{table}|{gas}|{sector}", drop=True)

    return res


def add_gas_totals(indf):
    # Should be called after aggregate_up_sectors

    top_level = [c for c in indf if "|" not in c]

    sector_stuff = combine_to_make_variable(
        indf.unstack("year").stack("sector", future_stack=True)
    )
    gas_totals = (
        indf[top_level]
        .sum(axis="columns")
        .unstack("year")
        .pix.format(variable="{table}|{gas}", drop=True)
    )

    res = pix.concat([sector_stuff, gas_totals])

    return res


@pytest.fixture(scope="session")
def example_raw_input():
    bottom_level_sectors = [
        # Aviation stuff
        "Energy|Demand|Transportation|Domestic Aviation",
        # something so we can get "Energy|Demand|Transportation" as a sum
        "Energy|Demand|Transportation|Rail",
        "Energy|Demand|Bunkers|International Aviation",
        # Industrial sector stuff
        "Energy|Demand|Industry",
        "Energy|Demand|Other Sector",
        "Industrial Processes",
        "Other",
        # Energy sector
        "Energy|Supply",
        # International shipping
        "Energy|Demand|Bunkers|International Shipping",
        # Residential commercial and other
        "Energy|Demand|Residential and Commercial and AFOFI",
        # Solvents production and application
        "Product Use",
        # Waste
        "Waste",
        # Agriculture
        "AFOLU|Agriculture",
        # Imperfect but put these in agriculture too for now
        # (we don't have a better source to harmonise too,
        # except for CO2 but they don't use CO2 LULUCF
        # emissions as input anyway, they use land-use change patterns).
        # (For SCMs, this also doesn't matter as it all gets rolled up to
        # "AFOLU" anyway because SCMs aren't able to handle the difference,
        # except maybe OSCAR but that isn't in OpenSCM-Runner
        # so we can't guess anything to do with OSCAR for now anyway.)
        "AFOLU|Land|Land Use and Land-Use Change",
        "AFOLU|Land|Harvested Wood Products",
        "AFOLU|Land|Other",
        "AFOLU|Land|Wetlands",
        # Burning sectors
        "AFOLU|Agricultural Waste Burning",
        "AFOLU|Land|Fires|Forest Burning",
        "AFOLU|Land|Fires|Grassland Burning",
        "AFOLU|Land|Fires|Peat Burning",
    ]

    timesteps = np.arange(2015, 2100 + 1, 5)
    start_index = pd.MultiIndex.from_product(
        [
            ["model_a"],
            ["scenario_a"],
            ["Emissions"],
            ["CO2", "CH4"],
            ["model_a|China", "model_a|Pacific OECD"],
            timesteps,
        ],
        names=["model", "scenario", "table", "gas", "region", "year"],
    )
    df = pd.DataFrame(
        RNG.random(timesteps.size * 4),
        columns=pd.Index(["Other"], name="sector"),
        index=start_index,
    )

    for bls in bottom_level_sectors:
        df[bls] = RNG.random(df.index.shape[0])

    df = aggregate_up_sectors(df)
    df = add_gas_totals(df)

    def get_unit(v):
        if "CO2" in v:
            return "Mt CO2/yr"
        if "CH4" in v:
            return "Mt CH4/yr"

        raise NotImplementedError(v)

    df["unit"] = df.index.get_level_values("variable").map(get_unit)
    df = df.set_index("unit", append=True)
    df = pix.concat(
        [
            df.groupby(df.index.names.difference(["region"]))
            .sum()
            .pix.assign(region="World"),
            df,
        ]
    )
    global_only_base = pd.DataFrame(
        RNG.random(timesteps.size)[np.newaxis, :],
        columns=df.columns,
        index=start_index.droplevel(["region", "year", "gas"]).drop_duplicates(),
    )
    global_only_l = []
    for global_only_gas, unit in [
        ("HFC|HFC23", "kt HFC23/yr"),
        ("HFC", "kt HFC134a-equiv/yr"),
        ("HFC|HFC134a", "kt HFC134a/yr"),
        ("HFC|HFC43-10", "kt HFC43-10/yr"),
        ("PFC", "kt CF4-equiv/yr"),
        ("F-Gases", "Mt CO2-equiv/yr"),
        ("SF6", "kt SF6/yr"),
        ("CF4", "kt CF4/yr"),
        ("C2F6", "kt C2F6/yr"),
        ("C6F14", "kt C6F14/yr"),
    ]:
        global_only_l.append(
            global_only_base.pix.assign(gas=global_only_gas, unit=unit, region="World")
            .pix.format(variable="{table}|{gas}", drop=True)
            .reorder_levels(df.index.names)
        )

    df = pix.concat([df, *global_only_l])

    return df


@pytest.fixture(scope="session")
def processed_output(example_raw_input):
    pre_processor = CMIP7ScenarioMIPPreProcessor(
        n_processes=None,  # run serially
    )

    processed = pre_processor(example_raw_input)

    yield processed


def test_transport_shuffling(example_raw_input, processed_output):
    """
    Test the moving of the transport data

    Domestic and international aviation should be added to make |Aircraft.
    Transportation should have domestic aviation
    subtracted to make |Transportation Sector.
    """
    # Not interested in global level for this
    df_to_check = processed_output.region_sector_workflow_emissions

    # The original Transportation should be dropped out
    assert (
        not df_to_check.pix.unique("variable")
        .str.endswith("Energy|Demand|Transportation")
        .any()
    ), df_to_check.pix.unique("variable")

    example_raw_input_sectors = split_variable(example_raw_input)

    exp_aircraft = combine_to_make_variable(
        example_raw_input_sectors.loc[
            pix.ismatch(
                sector=[
                    "Energy|Demand|Transportation|Domestic Aviation",
                    "Energy|Demand|Bunkers|International Aviation",
                ],
                gas=["CO2", "CH4"],
            )
        ]
        .groupby(example_raw_input_sectors.index.names.difference(["sector"]))
        .sum()
        .pix.assign(sector="Aircraft")
    ).reorder_levels(example_raw_input.index.names)

    example_raw_input_sectors_stacked = (
        example_raw_input_sectors.loc[pix.ismatch(gas=["CO2", "CH4"])]
        .unstack("sector")
        .stack("year", future_stack=True)
    )
    exp_transportation_sector = combine_to_make_variable(
        (
            example_raw_input_sectors_stacked["Energy|Demand|Transportation"]
            - example_raw_input_sectors_stacked[
                "Energy|Demand|Transportation|Domestic Aviation"
            ]
        )
        .unstack("year")
        .pix.assign(sector="Transportation Sector")
    ).reorder_levels(example_raw_input.index.names)

    assert_frame_equal(
        df_to_check.loc[pix.ismatch(variable="**Aircraft")],
        exp_aircraft,
    )

    assert_frame_equal(
        df_to_check.loc[pix.ismatch(variable="**Transportation Sector")],
        exp_transportation_sector,
    )


def test_industrial_sector_aggregation(example_raw_input, processed_output):
    df_to_check = processed_output.region_sector_workflow_emissions

    exp_sector = "Industrial Sector"
    exp_contributing_sectors = [
        "Energy|Supply",
        "Energy|Demand|Industry",
        "Energy|Demand|Other Sector",
        "Industrial Processes",
        "Other",
    ]

    example_raw_input_sectors = split_variable(example_raw_input)
    exp_sector_df = combine_to_make_variable(
        example_raw_input_sectors.loc[pix.ismatch(sector=exp_contributing_sectors)]
        .groupby(example_raw_input_sectors.index.names.difference(["sector"]))
        .sum()
        .pix.assign(sector=exp_sector)
    ).reorder_levels(example_raw_input.index.names)

    assert_frame_equal(
        df_to_check.loc[pix.ismatch(variable=f"**{exp_sector}")],
        exp_sector_df,
    )


def test_output_sectors(example_raw_input, processed_output):
    example_raw_input_sectors = split_variable(example_raw_input)

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
            ["CO2", "CH4"],
            exp_output_sectors,
        )
    ]

    assert set(exp_output_variables) == set(
        processed_output.region_sector_workflow_emissions.pix.unique("variable")
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
    and `processed_output.reaggregated_emissions`
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

    non_sector_group_levels = region_sector_split.index.names.difference(["sector"])
    region_sector_totals = region_sector_split.groupby(non_sector_group_levels).sum()

    region_sector_compare_non_co2 = region_sector_totals.loc[
        ~pix.isin(gas="CO2") & pix.isin(region="World")
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
        region_sector_split.loc[
            pix.isin(gas="CO2", region="World", sector=fossil_sectors)
        ]
        .groupby(non_sector_group_levels)
        .sum()
        .pix.assign(sector="Energy and Industrial Processes")
    )
    region_sector_compare_co2_afolu = combine_to_make_variable(
        region_sector_split.loc[
            pix.isin(gas="CO2", region="World") & ~pix.isin(sector=fossil_sectors)
        ]
        .groupby(non_sector_group_levels)
        .sum()
        .pix.assign(sector="AFOLU")
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


def test_output_region_sector_internal_consistency(processed_output):
    assert False, "check that sum over sectors makes sense"
    assert False, "check that sum over regions makes sense"
    assert False, "check that sum over sectors and regions makes sense"
    assert False, "check that sum over regions and sectors makes sense"


def test_input_not_internally_consistent_error(example_raw_input):
    # break the internal consistency at some level
    # make sure that a sensible error message is raised
    # (don't bother trying to guess what has gone wrong, too hard basket)
    assert False, "implement"


# Tests of missing key things for mapping

# Underlying logic:
# - we're doing region-sector harmonisation
# - hence we need regions and sectors lined up very specifically with CEDS
# - hence this kind of pre-processing
#   (if you want different pre-processing, use something more like AR6)
