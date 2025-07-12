"""
Handling of gridding emissions
"""

from __future__ import annotations

import itertools
import sys
from typing import TYPE_CHECKING

import pandas as pd
from pandas_openscm.grouping import groupby_except

from gcages.index_manipulation import (
    combine_sectors,
    combine_species,
    set_new_single_value_levels,
    split_sectors,
)

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

if TYPE_CHECKING:
    from gcages.typing import NP_FLOAT_OR_INT

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
"""
Complete set of species for gridding
"""

COMPLETE_GRIDDING_SECTORS: tuple[str, ...] = (
    "Agricultural Waste Burning",
    "Agriculture",
    "Aircraft",
    "BECCS",
    "Energy Sector",
    "Forest Burning",
    "Grassland Burning",
    "Industrial Sector",
    "International Shipping",
    "Other non-Land CDR",
    "Peat Burning",
    "Residential Commercial Other",
    "Solvents Production and Application",
    "Transportation Sector",
    "Waste",
)
"""
Complete set of sectors for gridding
"""


def get_complete_gridding_index(  # noqa: PLR0913
    model_regions: tuple[str, ...],
    world_gridding_sectors: tuple[str, ...] = (
        "Aircraft",
        "International Shipping",
    ),
    world_region: str = "World",
    region_level: str = "region",
    variable_level: str = "variable",
    table: tuple[str, ...] = (
        "Emissions",
        "Carbon Removal",
    ),
    level_separator: str = "|",
) -> pd.MultiIndex:
    """
    Get the index of complete gridding data

    Parameters
    ----------
    model_regions
        Model regions to use while reaggregating

    world_gridding_sectors
        Sectors that should only be gridded at the world level

    world_region
        The value used when the data represents the sum over all regions

    region_level
        Region level in the data index

    variable_level
        Variable level in the data index

    table
        Name of the 'table' for emissions

        Used to process and create variable names

    level_separator
        Separator between levels in the variable names

    Returns
    -------
    :
        Index of complete gridding data
    """
    complete_world_variables = [
        level_separator.join(["Emissions", species, sectors])
        for species, sectors in itertools.product(
            COMPLETE_GRIDDING_SPECIES, world_gridding_sectors
        )
    ]
    world_required = pd.MultiIndex.from_product(
        [complete_world_variables, [world_region]], names=[variable_level, region_level]
    )

    model_region_sectors = sorted(
        set(COMPLETE_GRIDDING_SECTORS) - set(world_gridding_sectors)
    )
    # Separating Emissions for CDR at regional level
    cdr_model_region_sectors = ["BECCS", "Other non-Land CDR"]
    emissions_model_region_sectors = [
        el for el in model_region_sectors if el not in cdr_model_region_sectors
    ]

    complete_model_region_variables = [
        level_separator.join(["Emissions", species, sectors])
        for species, sectors in itertools.product(
            COMPLETE_GRIDDING_SPECIES, emissions_model_region_sectors
        )
    ] + [
        level_separator.join(["Carbon Removal", species, sectors])
        for species, sectors in itertools.product(["CO2"], cdr_model_region_sectors)
    ]

    model_region_required = pd.MultiIndex.from_product(
        [complete_model_region_variables, model_regions],
        names=[variable_level, region_level],
    )

    res: pd.MultiIndex = world_required.append(model_region_required)  # type: ignore # pandas-stubs out of date

    return res


class SpatialResolutionOption(StrEnum):
    """Spatial resolution option"""

    WORLD = "world"
    """Data reported at the world (i.e. global) level"""

    MODEL_REGION = "model_region"
    """Data reported at the (IAM) model region level"""


CO2_FOSSIL_SECTORS_GRIDDING: tuple[str, ...] = (
    "Aircraft",
    "BECCS",
    "International Shipping",
    "Energy Sector",
    "Industrial Sector",
    "Other non-Land CDR",
    "Residential Commercial Other",
    "Solvents Production and Application",
    "Transportation Sector",
    "Waste",
)
"""
Sectors that come from or go to fossil CO2 reservoirs (gridding naming convention)

BECCS is here because the carbon is stored permanently (or assumed to be).
It is grown then removed from the land pool,
so is 'net zero' from the land pool's point of view
(and handling this really well requires running a carbon cycle model
to determine the possible uptake from the BECCS land-use,
which isn't how the split between modelling domains works at the moment).

There is the same issue for some non-land CDR e.g. ocean alkalinity stuff.
Again, a handling sophisticiated enough to capture this properly
is beyond the scope of the fossil/biosphere split we're making here.

Not a perfect split with [CO2_BIOSPHERE_SECTORS_GRIDDING][(m).],
but the best we can do.
"""

CO2_BIOSPHERE_SECTORS_GRIDDING: tuple[str, ...] = (
    # Agriculture in biosphere because most of its emissions
    # are land carbon cycle (but not all, probably, in reality)
    "Agriculture",
    "Agricultural Waste Burning",
    "Forest Burning",
    "Grassland Burning",
    "Peat Burning",
)
"""
Sectors that come from biospheric CO2 reservoirs (gridding naming convention)

Not a perfect split with [CO2_FOSSIL_SECTORS_GRIDDING][(m).],
but the best we can do.
"""


def to_global_workflow_emissions(  # noqa: PLR0913
    gridding_emissions: pd.DataFrame,
    time_name: str = "year",
    region_level: str = "region",
    world_region: str = "World",
    global_workflow_co2_fossil_sector: str = "Fossil",
    global_workflow_co2_biosphere_sector: str = "Biosphere",
    co2_fossil_sectors: tuple[str, ...] = CO2_FOSSIL_SECTORS_GRIDDING,
    co2_biosphere_sectors: tuple[str, ...] = CO2_BIOSPHERE_SECTORS_GRIDDING,
    sectors_level: str = "sectors",
    species_level: str = "species",
    co2_name: str = "CO2",
) -> pd.DataFrame:
    """
    Convert gridding emissions to global workflow emissions

    Parameters
    ----------
    gridding_emissions
        Gridding emissions

    time_name
        Name of the time axis in `gridding_emissions`

    region_level
        Region level in the data index

    world_region
        The value used when the data represents the sum over all regions

    global_workflow_co2_fossil_sector
        Name of the CO2 'sector' with fossil origins to use in the output

    global_workflow_co2_biosphere_sector
        Name of the CO2 'sector' with biospheric origins to use in the output

    co2_fossil_sectors
        Sectors to assume have an origin in fossil CO2 reservoirs

    co2_biosphere_sectors
        Sectors to assume have an origin in biospheric CO2 reservoirs

    sectors_level
        Sectors level in the data index

    species_level
        Species level in the data index

    co2_name
        String that indicates emissions of CO2 in variable names

    Returns
    -------
    :
        Global workflow emissions
    """
    stacked: pd.DataFrame = (
        split_sectors(  # type: ignore
            gridding_emissions,
            middle_level=species_level,
            bottom_level=sectors_level,
        )
        .stack()
        .unstack("sectors")
    )

    world_locator = stacked.index.get_level_values(region_level) == world_region
    region_sector_df = stacked.loc[~world_locator]
    sector_df = stacked.loc[world_locator].reset_index("region", drop=True)

    gw_sector_df, gw_total_df = to_global_workflow_emissions_from_stacked(
        region_sector_df=region_sector_df,
        sector_df=sector_df,
        time_name=time_name,
        region_level=region_level,
        global_workflow_co2_fossil_sector=global_workflow_co2_fossil_sector,
        global_workflow_co2_biosphere_sector=global_workflow_co2_biosphere_sector,
        co2_fossil_sectors=co2_fossil_sectors,
        co2_biosphere_sectors=co2_biosphere_sectors,
        sectors_level=sectors_level,
        species_level=species_level,
        co2_name=co2_name,
    )

    gw_sector_df_input_like = set_new_single_value_levels(
        combine_sectors(
            gw_sector_df,  # type: ignore # fix when moving to pandas-openscm
            middle_level=species_level,
            bottom_level=sectors_level,
        ),
        {region_level: world_region},
    ).unstack(time_name)
    gw_total_df_input_like = set_new_single_value_levels(
        combine_species(gw_total_df, bottom_level=species_level),  # type: ignore # fix when moving to pandas-openscm
        {region_level: world_region},
    ).unstack(time_name)

    res = pd.concat(
        [
            df.reorder_levels(gridding_emissions.index.names)
            for df in [gw_total_df_input_like, gw_sector_df_input_like]
        ]
    )
    return res


def to_global_workflow_emissions_from_stacked(  # noqa: PLR0913
    region_sector_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    time_name: str,
    region_level: str,
    global_workflow_co2_fossil_sector: str,
    global_workflow_co2_biosphere_sector: str,
    co2_fossil_sectors: tuple[str, ...],
    co2_biosphere_sectors: tuple[str, ...],
    sectors_level: str,
    species_level: str,
    co2_name: str,
) -> tuple[pd.Series[NP_FLOAT_OR_INT], pd.Series[NP_FLOAT_OR_INT]]:  # type: ignore # pandas-stubs out of date
    """
    Convert pre-stacked gridding emissions to global workflow emissions

    Parameters
    ----------
    region_sector_df
        Data with region and sector levels

    sector_df
        Data with sector levels only

    time_name
        Name of the time axis in `gridding_emissions`

    region_level
        Region level in the data index

    global_workflow_co2_fossil_sector
        Name of the CO2 'sector' with fossil origins to use in the output

    global_workflow_co2_biosphere_sector
        Name of the CO2 'sector' with biospheric origins to use in the output

    co2_fossil_sectors
        Sectors to assume have an origin in fossil CO2 reservoirs

    co2_biosphere_sectors
        Sectors to assume have an origin in biospheric CO2 reservoirs

    sectors_level
        Sectors level in the data index

    species_level
        Species level in the data index

    co2_name
        String that indicates emissions of CO2 in variable names

    Returns
    -------
    sectors
        Global workflow emissions with a sector level

    totals
        Global workflow emissions only with totals (no region or sector level)
    """
    region_sector_df_region_sum = groupby_except(region_sector_df, region_level).sum()

    sector_df_full = pd.concat([sector_df, region_sector_df_region_sum], axis="columns")

    co2_locator = (sector_df_full.index.get_level_values(species_level) == co2_name) & (
        sector_df_full.index.get_level_values("table") == "Emissions"
    )

    non_co2: pd.Series[NP_FLOAT_OR_INT] = sector_df_full[~co2_locator].sum(  # type: ignore # pandas-stubs out of date
        axis="columns"
    )

    not_used_cols = sorted(
        set(sector_df_full.columns)
        - {
            *co2_biosphere_sectors,
            *co2_fossil_sectors,
        }
    )
    if not_used_cols:
        msg = (
            "For the given inputs, not all CO2 sectors will be used.\n"
            f"{not_used_cols=}\n"
            f"{co2_fossil_sectors=}\n"
            f"{co2_biosphere_sectors=}\n"
        )
        raise AssertionError(msg)

    co2_fossil = set_new_single_value_levels(
        sector_df_full.loc[co2_locator, list(co2_fossil_sectors)].sum(axis="columns"),
        {sectors_level: global_workflow_co2_fossil_sector},
    )
    co2_biosphere = set_new_single_value_levels(
        sector_df_full.loc[co2_locator, list(co2_biosphere_sectors)].sum(
            axis="columns"
        ),
        {sectors_level: global_workflow_co2_biosphere_sector},
    )

    totals = non_co2
    sectors: pd.Series[NP_FLOAT_OR_INT] = pd.concat(  # type: ignore # pandas-stubs out of date
        [
            df.reorder_levels(co2_fossil.index.names)
            for df in [co2_fossil, co2_biosphere]
        ]
    )

    return sectors, totals
