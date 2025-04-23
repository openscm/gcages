"""
Operations that assume the data has the sectors in the columns
"""

from __future__ import annotations

import pandas as pd
from pandas_openscm.grouping import groupby_except

from gcages.cmip7_scenariomip.pre_processing.constants import (
    AVIATION_SECTOR_REAGGREGATED,
    DOMESTIC_AVIATION_SECTOR_INPUT,
    INTERNATIONAL_AVIATION_SECTOR_INPUT,
    TRANSPORTATION_SECTOR_INPUT,
    TRANSPORTATION_SECTOR_REAGGREGATED,
)
from gcages.index_manipulation import (
    combine_sectors,
    combine_species,
    set_new_single_value_levels,
)


def reclassify_aviation_emissions(
    region_sector_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    region_level: str,
    domestic_aviation_sector: str = DOMESTIC_AVIATION_SECTOR_INPUT,
    international_aviation_sector: str = INTERNATIONAL_AVIATION_SECTOR_INPUT,
    transportation_sector: str = TRANSPORTATION_SECTOR_INPUT,
    aviation_sector_reaggregated: str = AVIATION_SECTOR_REAGGREGATED,
    transportation_sector_reaggregated: str = TRANSPORTATION_SECTOR_REAGGREGATED,
    copy: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if copy:
        region_sector_df = region_sector_df.copy()
        sector_df = sector_df.copy()

    # Will need some if clause in here
    # if domestic aviation is only reported at World level
    domestic_aviation_sum = groupby_except(
        region_sector_df[domestic_aviation_sector], region_level
    ).sum()

    sector_df[aviation_sector_reaggregated] = (
        sector_df[international_aviation_sector] + domestic_aviation_sum
    )
    region_sector_df[transportation_sector_reaggregated] = (
        region_sector_df[transportation_sector]
        - region_sector_df[domestic_aviation_sector]
    )

    # Drop out sectors we're not using
    sector_df = sector_df.drop([international_aviation_sector], axis="columns")
    region_sector_df = region_sector_df.drop(
        [domestic_aviation_sector, transportation_sector], axis="columns"
    )

    return region_sector_df, sector_df


def aggregate_sector(
    indf: pd.DataFrame,
    sector_out: str,
    sector_components: list[str],
    time_name: str,
    allow_missing: list[str] | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    if copy:
        indf = indf.copy()

    to_sum = sector_components
    if allow_missing is not None:
        missing = {c for c in to_sum if c not in sector_components}
        # Anything which is missing and allowed to be missing,
        # we can drop from to_sum
        to_drop_from_sum = missing.intersection(set(allow_missing))
        to_sum = list(set(to_sum) - to_drop_from_sum)

    indf[sector_out] = indf[to_sum].sum(axis="columns", min_count=len(to_sum))
    indf = indf.drop(to_sum, axis="columns")

    return indf


def gridding_emissions_to_global_workflow_emissions(
    region_sector_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    time_name: str,
    region_level: str,
    world_region: str,
    global_workflow_co2_fossil_sector: str,
    global_workflow_co2_biosphere_sector: str,
    co2_fossil_sectors: tuple[str, ...],
    co2_biosphere_sectors: tuple[str, ...],
    species_level: str,
    co2_name: str,
) -> pd.DataFrame:
    region_sector_df_region_sum = groupby_except(region_sector_df, region_level).sum()

    sector_df_full = pd.concat([sector_df, region_sector_df_region_sum], axis="columns")

    co2_locator = sector_df_full.index.get_level_values(species_level) == co2_name

    non_co2 = combine_species(
        set_new_single_value_levels(
            sector_df_full[~co2_locator].sum(axis="columns"),
            {region_level: world_region},
        )
    )

    not_used_cols = set(sector_df_full.columns) - {
        *co2_biosphere_sectors,
        *co2_fossil_sectors,
    }
    if not_used_cols:
        raise AssertionError(not_used_cols)

    co2_fossil = combine_sectors(
        set_new_single_value_levels(
            sector_df_full.loc[co2_locator, list(co2_fossil_sectors)].sum(
                axis="columns"
            ),
            {region_level: world_region, "sectors": global_workflow_co2_fossil_sector},
        )
    )
    co2_biosphere = combine_sectors(
        set_new_single_value_levels(
            sector_df_full.loc[co2_locator, list(co2_biosphere_sectors)].sum(
                axis="columns"
            ),
            {
                region_level: world_region,
                "sectors": global_workflow_co2_biosphere_sector,
            },
        )
    )

    res = pd.concat(
        [
            s.reorder_levels(non_co2.index.names)
            for s in [non_co2, co2_fossil, co2_biosphere]
        ]
    ).unstack(time_name)

    return res
