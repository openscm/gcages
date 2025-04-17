"""
Integration tests of our pre-processing for CMIP7 ScenarioMIP
"""

import numpy as np
import pandas as pd
import pytest

from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
from gcages.testing import assert_frame_equal

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
    df = pd.DataFrame(
        RNG.random(timesteps.size * 4),
        columns=pd.Index(["Other"], name="sector"),
        index=pd.MultiIndex.from_product(
            [
                ["model_a"],
                ["scenario_a"],
                ["Emissions"],
                ["CO2", "CH4"],
                ["model_a|China", "model_a|Pacific OECD"],
                timesteps,
            ],
            names=["model", "scenario", "table", "gas", "region", "year"],
        ),
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
                ]
            )
        ]
        .groupby(example_raw_input_sectors.index.names.difference(["sector"]))
        .sum()
        .pix.assign(sector="Aircraft")
    ).reorder_levels(example_raw_input.index.names)

    restacked = example_raw_input_sectors.unstack("sector").stack(
        "year", future_stack=True
    )
    exp_transportation_sector = combine_to_make_variable(
        (
            restacked["Energy|Demand|Transportation"]
            - restacked["Energy|Demand|Transportation|Domestic Aviation"]
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
    exp_sector = "Industrial Sector"
    exp_contributing_sectors = [
        "Energy|Supply",
        "Energy|Demand|Industry",
        "Energy|Demand|Other Sector",
        "Industrial Processes",
        "Other",
    ]
    assert False, "Implement"


def test_output_sectors(processed_output):
    exp_output_sectors = [
        "Energy Sector",
        "International Shipping",
        "Residential Commercial Other"
        "Solvents Production and Application"
        "Agriculture"
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Peat Burning",
    ]
    assert False, "Implement"


def test_output_internal_consistency(processed_output):
    assert (
        False
    ), "check that global workflow output is consistent with region-sector outpu"


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


# Underlying logic:
# - we're doing region-sector harmonisation
# - hence we need regions and sectors lined up very specifically with CEDS
# - hence this kind of pre-processing
#   (if you want different pre-processing, use something more like AR6)
