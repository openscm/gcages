"""
Tests of the `gcages.cmip7_scenariomip.pre_processing.gridding_emissions`
"""

from __future__ import annotations

import itertools
import re
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from pandas_openscm.grouping import groupby_except

from gcages.cmip7_scenariomip.gridding_emissions import to_global_workflow_emissions
from gcages.index_manipulation import (
    combine_sectors,
    combine_species,
    create_levels_based_on_existing,
    set_new_single_value_levels,
    split_sectors,
)
from gcages.testing import assert_frame_equal, get_variable_unit_default
from gcages.typing import NP_ARRAY_OF_FLOAT_OR_INT

pytest.importorskip("pandas_indexing")

RNG = np.random.default_rng()

COMPLETE_GRIDDING_SPECIES: tuple[str, ...] = (
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
)

COMPLETE_GRIDDING_SECTORS_WORLD: tuple[str, ...] = (
    "Aircraft",
    "International Shipping",
)

COMPLETE_GRIDDING_SECTORS_MODEL_REGION: tuple[str, ...] = (
    "Agriculture",
    "Agricultural Waste Burning",
    "Transportation Sector",
    "Energy Sector",
    "Forest Burning",
    "Grassland Burning",
    "Industrial Sector",
    "International Shipping",
    "Peat Burning",
    "Residential Commercial Other",
    "Solvents Production and Application",
    "Waste",
    "BECCS",
    "Other non-Land CDR",
)

# For most of the tests, use the same world and model regions.
# Can obviously parameterise differently in individual tests.
WORLD_REGION = "World"
MODEL = "model_a"
MODEL_REGIONS = [f"{MODEL}|{r}" for r in ["Pacific OECD", "China"]]


def get_gridding_emissions(  # noqa: PLR0913
    world_region: str = WORLD_REGION,
    model_regions: list[str] = MODEL_REGIONS,
    timepoints: NP_ARRAY_OF_FLOAT_OR_INT = np.arange(2010, 2100 + 1, 10.0),
    columns_name: str = "year",
    model: str = MODEL,
    scenario: str = "scenario_a",
    model_level: str = "model",
    scenario_level: str = "scenario",
    get_variable_unit: Callable[[str], str] = get_variable_unit_default,
):
    variables_world = [
        f"Emissions|{species}|{sector}"
        for species, sector in itertools.product(
            COMPLETE_GRIDDING_SPECIES, COMPLETE_GRIDDING_SECTORS_WORLD
        )
    ]
    variables_model_region = [
        f"Emissions|{species}|{sector}"
        for species, sector in itertools.product(
            COMPLETE_GRIDDING_SPECIES, COMPLETE_GRIDDING_SECTORS_MODEL_REGION
        )
    ]

    index_world = pd.MultiIndex.from_product(
        [variables_world, [world_region]], names=["variable", "region"]
    )
    index_model_region = pd.MultiIndex.from_product(
        [variables_model_region, model_regions], names=["variable", "region"]
    )

    index = index_world.append(index_model_region)
    index = create_levels_based_on_existing(
        index, {"unit": ("variable", get_variable_unit)}
    )

    res = set_new_single_value_levels(
        pd.DataFrame(
            RNG.random((index.shape[0], timepoints.size)),
            columns=pd.Index(timepoints, name=columns_name),
            index=index,
        ),
        {model_level: model, scenario_level: scenario},
    )

    return res


@pytest.fixture
def gridding_emissions():
    return get_gridding_emissions()


@pytest.mark.parametrize(
    "res_co2_fossil_name, res_co2_fossil_name_exp",
    (
        (None, "Fossil"),
        ("Energy and Industrial Processes", "Energy and Industrial Processes"),
    ),
)
@pytest.mark.parametrize(
    "res_co2_biosphere_name, res_co2_biosphere_name_exp",
    (
        (None, "Biosphere"),
        ("AFOLU", "AFOLU"),
    ),
)
@pytest.mark.parametrize(
    [
        "co2_fossil_sectors",
        "co2_fossil_sectors_exp",
        "co2_biosphere_sectors",
        "co2_biosphere_sectors_exp",
    ],
    (
        pytest.param(
            None,
            (
                "Aircraft",
                "International Shipping",
                "Energy Sector",
                "Industrial Sector",
                "Residential Commercial Other",
                "Solvents Production and Application",
                "Transportation Sector",
                "Waste",
                "BECCS",
                "Other non-Land CDR",
            ),
            None,
            (
                "Agriculture",
                "Agricultural Waste Burning",
                "Forest Burning",
                "Grassland Burning",
                "Peat Burning",
            ),
            id="default",
        ),
        pytest.param(
            (
                "Aircraft",
                "International Shipping",
                "Energy Sector",
                "Industrial Sector",
                "Residential Commercial Other",
                "Solvents Production and Application",
                "Transportation Sector",
                "Waste",
                "BECCS",
                "Other non-Land CDR",
                "Agriculture",
            ),
            (
                "Aircraft",
                "International Shipping",
                "Energy Sector",
                "Industrial Sector",
                "Residential Commercial Other",
                "Solvents Production and Application",
                "Transportation Sector",
                "Waste",
                "BECCS",
                "Other non-Land CDR",
                "Agriculture",
            ),
            (
                "Agricultural Waste Burning",
                "Forest Burning",
                "Grassland Burning",
                "Peat Burning",
            ),
            (
                "Agricultural Waste Burning",
                "Forest Burning",
                "Grassland Burning",
                "Peat Burning",
            ),
            id="agriculture-in-fossil",
        ),
    ),
)
def test_to_global_workflow_emissions(  # noqa: PLR0913
    res_co2_fossil_name,
    res_co2_fossil_name_exp,
    res_co2_biosphere_name,
    res_co2_biosphere_name_exp,
    co2_fossil_sectors,
    co2_fossil_sectors_exp,
    co2_biosphere_sectors,
    co2_biosphere_sectors_exp,
    gridding_emissions,
):
    kwargs = {}
    if res_co2_fossil_name is not None:
        kwargs["global_workflow_co2_fossil_sector"] = res_co2_fossil_name

    if res_co2_biosphere_name is not None:
        kwargs["global_workflow_co2_biosphere_sector"] = res_co2_biosphere_name

    if co2_fossil_sectors is not None:
        kwargs["co2_fossil_sectors"] = co2_fossil_sectors

    if co2_biosphere_sectors is not None:
        kwargs["co2_biosphere_sectors"] = co2_biosphere_sectors

    res = to_global_workflow_emissions(gridding_emissions, **kwargs)

    tmp_split = split_sectors(gridding_emissions)

    co2_locator = tmp_split.index.get_level_values("species") == "CO2"
    non_co2_exp = set_new_single_value_levels(
        combine_species(
            groupby_except(
                tmp_split.loc[~co2_locator],
                ["region", "sectors"],
            ).sum()
        ),
        {"region": "World"},
    )

    co2_exp_l = []
    for name, components in (
        (res_co2_fossil_name_exp, co2_fossil_sectors_exp),
        (res_co2_biosphere_name_exp, co2_biosphere_sectors_exp),
    ):
        co2_global_sector = combine_sectors(
            set_new_single_value_levels(
                groupby_except(
                    tmp_split.loc[
                        co2_locator
                        & tmp_split.index.get_level_values("sectors").isin(components)
                    ],
                    ["region", "sectors"],
                ).sum(),
                {"region": "World", "sectors": name},
            )
        )
        co2_exp_l.append(co2_global_sector)

    exp = pd.concat(
        [
            df.reorder_levels(gridding_emissions.index.names)
            for df in [non_co2_exp, *co2_exp_l]
        ]
    )

    assert_frame_equal(res, exp)


def test_to_global_workflow_emissions_missing_sector_error(gridding_emissions):
    co2_fossil_sectors = (
        "Aircraft",
        "BECCS",
        "International Shipping",
        "Energy Sector",
        "Industrial Sector",
        "Other non-Land CDR",
        "Residential Commercial Other",
        # "Solvents Production and Application",
        "Transportation Sector",
        "Waste",
    )

    co2_biosphere_sectors = (
        "Agriculture",
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Peat Burning",
    )

    not_used_cols = sorted(
        [
            "Solvents Production and Application",
        ]
    )
    error_msg = re.escape(
        "\n".join(
            [
                "For the given inputs, not all CO2 sectors will be used.",
                f"{not_used_cols=}",
                f"{co2_fossil_sectors=}",
                f"{co2_biosphere_sectors=}",
            ]
        )
    )

    with pytest.raises(AssertionError, match=error_msg):
        to_global_workflow_emissions(
            gridding_emissions,
            co2_fossil_sectors=co2_fossil_sectors,
            co2_biosphere_sectors=co2_biosphere_sectors,
        )
