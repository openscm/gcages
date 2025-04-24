"""
Basic reaggregation

This is called 'basic' because it's the first one we thought about.
It's also, in some ways, the simplest.
It assumes that domestic aviation is reported at the model region level.
"""

from __future__ import annotations

import sys

import pandas as pd
from attrs import define

from gcages.completeness import assert_all_groups_are_complete

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

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


class SpatialResolutionOption(StrEnum):
    """Spatial resolution option"""

    WORLD = "world"
    """Data reported at the world (i.e. global) level"""

    MODEL_REGION = "model_region"
    """Data reported at the (IAM) model region level"""


@define
class GriddingSectorComponentsReporting:
    """
    Definition of the components of a gridding sector for reporting

    OR logic is applied to the exclusions
    i.e. a variable will not be required
    if the sector is in `input_sectors_optional`
    or the species is in `input_species_optional`
    (i.e. we are maximally relaxed about optional reporting,
    instead of using AND logic and being restrictive).
    """

    gridding_sector: str
    """The gridding sector"""

    spatial_resolution: SpatialResolutionOption

    input_sectors: tuple[str, ...]
    """The input sectors"""

    input_sectors_optional: tuple[str, ...]
    """The input sectors that are optional"""

    input_species_optional: tuple[str, ...]
    """The input species that are optional"""

    def to_complete_variables(self, all_species: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            f"Emissions|{species}|{sector}"
            for species in all_species
            for sector in self.input_sectors
        )

    def to_required_variables(self, all_species: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            f"Emissions|{species}|{sector}"
            for species in all_species
            for sector in self.input_sectors
            if not (
                sector in self.input_sectors_optional
                or species in self.input_species_optional
            )
        )


gridding_sectors = (
    GriddingSectorComponentsReporting(
        gridding_sector="Agriculture",
        spatial_resolution=SpatialResolutionOption.MODEL_REGION,
        input_sectors=("AFOLU|Agriculture",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Agricultural Waste Burning",
        spatial_resolution=SpatialResolutionOption.MODEL_REGION,
        input_sectors=(
            "AFOLU|Agricultural Waste Burning",
            "AFOLU|Land|Harvested Wood Products",
            "AFOLU|Land|Land Use and Land-Use Change",
            "AFOLU|Land|Other",
            "AFOLU|Land|Wetlands",
        ),
        input_sectors_optional=(
            "AFOLU|Land|Harvested Wood Products",
            "AFOLU|Land|Land Use and Land-Use Change",
            "AFOLU|Land|Other",
            "AFOLU|Land|Wetlands",
        ),
        input_species_optional=(
            "BC",
            "CO",
            "OC",
            "Sulfur",
        ),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Aircraft",
        spatial_resolution=SpatialResolutionOption.WORLD,
        input_sectors=(
            "Energy|Demand|Bunkers|International Aviation",
            # Domestic aviation is included too.
            # However, it has to be reported at the regional level
            # so we can subtract it from Transport
            # (hence it doesn't appear here, see below)
        ),
        input_sectors_optional=(),
        input_species_optional=("CH4",),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Domestic aviation headache",
        spatial_resolution=SpatialResolutionOption.MODEL_REGION,
        input_sectors=("Energy|Demand|Transportation|Domestic Aviation",),
        input_sectors_optional=(),
        input_species_optional=("CH4",),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Transportation Sector",
        spatial_resolution=SpatialResolutionOption.MODEL_REGION,
        input_sectors=("Energy|Demand|Transportation",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Energy Sector",
        spatial_resolution=SpatialResolutionOption.MODEL_REGION,
        input_sectors=("Energy|Supply",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Forest Burning",
        spatial_resolution=SpatialResolutionOption.MODEL_REGION,
        input_sectors=("AFOLU|Land|Fires|Forest Burning",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Grassland Burning",
        spatial_resolution=SpatialResolutionOption.MODEL_REGION,
        input_sectors=("AFOLU|Land|Fires|Grassland Burning",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Industrial Sector",
        spatial_resolution=SpatialResolutionOption.MODEL_REGION,
        input_sectors=(
            "Energy|Demand|Industry",
            "Energy|Demand|Other Sector",
            "Industrial Processes",
            "Other",
            "Other Capture and Removal",
        ),
        input_sectors_optional=(
            "Energy|Demand|Other Sector",
            "Other",
            "Other Capture and Removal",
        ),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="International Shipping",
        spatial_resolution=SpatialResolutionOption.WORLD,
        input_sectors=("Energy|Demand|Bunkers|International Shipping",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Peat Burning",
        spatial_resolution=SpatialResolutionOption.MODEL_REGION,
        input_sectors=("AFOLU|Land|Fires|Peat Burning",),
        input_sectors_optional=("AFOLU|Land|Fires|Peat Burning",),
        input_species_optional=COMPLETE_GRIDDING_SPECIES,
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Residential Commercial Other",
        spatial_resolution=SpatialResolutionOption.MODEL_REGION,
        input_sectors=("Energy|Demand|Residential and Commercial and AFOFI",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Solvents Production and Application",
        spatial_resolution=SpatialResolutionOption.MODEL_REGION,
        input_sectors=("Product Use",),
        input_sectors_optional=(),
        input_species_optional=("BC", "CH4", "CO", "NOx", "OC", "Sulfur"),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Waste",
        spatial_resolution=SpatialResolutionOption.MODEL_REGION,
        input_sectors=("Waste",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
)

COMPLETE_WORLD_VARIABLES: tuple[str, ...] = tuple(
    v
    for gs in gridding_sectors
    if gs.spatial_resolution == SpatialResolutionOption.WORLD
    for v in gs.to_complete_variables(all_species=COMPLETE_GRIDDING_SPECIES)
)
"""
Complete set of variables at the world level
"""

REQUIRED_WORLD_VARIABLES: tuple[str, ...] = tuple(
    v
    for gs in gridding_sectors
    if gs.spatial_resolution == SpatialResolutionOption.WORLD
    for v in gs.to_required_variables(all_species=COMPLETE_GRIDDING_SPECIES)
)
"""
Required set of variables at the world level
"""

OPTIONAL_WORLD_VARIABLES: tuple[str, ...] = tuple(
    set(COMPLETE_WORLD_VARIABLES) - set(REQUIRED_WORLD_VARIABLES)
)
"""
Optional set of variables at the world level
"""

COMPLETE_MODEL_REGION_VARIABLES: tuple[str, ...] = tuple(
    v
    for gs in gridding_sectors
    if gs.spatial_resolution == SpatialResolutionOption.MODEL_REGION
    for v in gs.to_complete_variables(all_species=COMPLETE_GRIDDING_SPECIES)
)
"""
Complete set of variables at the model region level
"""

REQUIRED_MODEL_REGION_VARIABLES: tuple[str, ...] = tuple(
    v
    for gs in gridding_sectors
    if gs.spatial_resolution == SpatialResolutionOption.MODEL_REGION
    for v in gs.to_required_variables(all_species=COMPLETE_GRIDDING_SPECIES)
)
"""
Required set of variables at the model region level
"""

OPTIONAL_MODEL_REGION_VARIABLES: tuple[str, ...] = tuple(
    set(COMPLETE_MODEL_REGION_VARIABLES) - set(REQUIRED_MODEL_REGION_VARIABLES)
)
"""
Optional set of variables at the model region level
"""


def get_complete_timeseries_index(
    model_regions: tuple[str, ...],
    world_region: str = "World",
    region_level: str = "region",
    variable_level: str = "variable",
) -> pd.MultiIndex:
    world_required = pd.MultiIndex.from_product(
        [COMPLETE_WORLD_VARIABLES, [world_region]], names=[variable_level, region_level]
    )

    model_region_required = pd.MultiIndex.from_product(
        [COMPLETE_MODEL_REGION_VARIABLES, model_regions],
        names=[variable_level, region_level],
    )

    res = world_required.append(model_region_required)

    return res


def get_required_timeseries_index(
    model_regions: tuple[str, ...],
    world_region: str = "World",
    region_level: str = "region",
    variable_level: str = "variable",
) -> pd.MultiIndex:
    world_required = pd.MultiIndex.from_product(
        [REQUIRED_WORLD_VARIABLES, [world_region]], names=[variable_level, region_level]
    )

    model_region_required = pd.MultiIndex.from_product(
        [REQUIRED_MODEL_REGION_VARIABLES, model_regions],
        names=[variable_level, region_level],
    )

    res = world_required.append(model_region_required)

    return res


def has_all_required_timeseries(
    df: pd.DataFrame,
    model_regions: tuple[str, ...],
) -> None:
    # Don't bother with kwargs etc. because this is a facade
    assert_all_groups_are_complete(
        df,
        get_required_timeseries_index(
            model_regions=model_regions,
        ),
    )
