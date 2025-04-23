"""
Conversion from gridding emissions to emissions used in the global workflow
"""

from __future__ import annotations

import pandas as pd

import gcages.cmip7_scenariomip.pre_processing.sector_cols
from gcages.cmip7_scenariomip.pre_processing.constants import (
    CO2_BIOSPHERE_SECTORS_GRIDDING,
    CO2_FOSSIL_SECTORS_GRIDDING,
)
from gcages.index_manipulation import split_sectors


def convert_gridding_emissions_to_global_workflow_emissions(  # noqa: PLR0913
    gridding_emissions: pd.DataFrame,
    time_name: str = "year",
    region_level: str = "region",
    world_region: str = "World",
    global_workflow_co2_fossil_sector: str = "Fossil",
    global_workflow_co2_biosphere_sector: str = "Biosphere",
    co2_fossil_sectors: tuple[str, ...] = CO2_FOSSIL_SECTORS_GRIDDING,
    co2_biosphere_sectors: tuple[str, ...] = CO2_BIOSPHERE_SECTORS_GRIDDING,
    species_level: str = "species",
    co2_name: str = "CO2",
) -> pd.DataFrame:
    stacked = (
        split_sectors(
            gridding_emissions,
            middle_level=species_level,
            bottom_level="sectors",
        )
        .stack()
        .unstack("sectors")
    )

    world_locator = stacked.index.get_level_values(region_level) == world_region
    region_sector_df = stacked.loc[~world_locator]
    sector_df = stacked.loc[world_locator].reset_index("region", drop=True)

    res = gcages.cmip7_scenariomip.pre_processing.sector_cols.convert_to_global_workflow_emissions(  # noqa: E501
        region_sector_df=region_sector_df,
        sector_df=sector_df,
        time_name=time_name,
        region_level=region_level,
        world_region=world_region,
        global_workflow_co2_fossil_sector=global_workflow_co2_fossil_sector,
        global_workflow_co2_biosphere_sector=global_workflow_co2_biosphere_sector,
        co2_fossil_sectors=co2_fossil_sectors,
        co2_biosphere_sectors=co2_biosphere_sectors,
        species_level=species_level,
        co2_name=co2_name,
    )

    return res
