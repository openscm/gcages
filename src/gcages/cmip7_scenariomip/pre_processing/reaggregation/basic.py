"""
Basic reaggregation

This is called 'basic' because it's the first one we thought about.
It's also, in some ways, the simplest.
It assumes that domestic aviation is reported at the model region level.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from attrs import define
from pandas_openscm.grouping import groupby_except
from pandas_openscm.indexing import multi_index_lookup, multi_index_match

from gcages.assertions import assert_only_working_on_variable_unit_region_variations
from gcages.completeness import assert_all_groups_are_complete, get_missing_levels
from gcages.index_manipulation import (
    combine_sectors,
    combine_species,
    create_levels_based_on_existing,
    set_new_single_value_levels,
    split_sectors,
)
from gcages.internal_consistency import InternalConsistencyError
from gcages.testing import compare_close

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

if TYPE_CHECKING:
    from gcages.typing import PINT_SCALAR

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


SECTOR_DOMESTIC_AVIATION = "Energy|Demand|Transportation|Domestic Aviation"
"""
Domestic aviation sector
"""

gridding_sectors_reporting = (
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
        input_sectors=(SECTOR_DOMESTIC_AVIATION,),
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
    for gs in gridding_sectors_reporting
    if gs.spatial_resolution == SpatialResolutionOption.WORLD
    for v in gs.to_complete_variables(all_species=COMPLETE_GRIDDING_SPECIES)
)
"""
Complete set of variables at the world level
"""

REQUIRED_WORLD_VARIABLES: tuple[str, ...] = tuple(
    v
    for gs in gridding_sectors_reporting
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
    for gs in gridding_sectors_reporting
    if gs.spatial_resolution == SpatialResolutionOption.MODEL_REGION
    for v in gs.to_complete_variables(all_species=COMPLETE_GRIDDING_SPECIES)
)
"""
Complete set of variables at the model region level
"""

REQUIRED_MODEL_REGION_VARIABLES: tuple[str, ...] = tuple(
    v
    for gs in gridding_sectors_reporting
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


def get_internal_consistency_checking_index(
    model_regions: tuple[str, ...],
    world_region: str = "World",
    region_level: str = "region",
    variable_level: str = "variable",
) -> pd.MultiIndex:
    world_internal_consistency_checking = pd.MultiIndex.from_product(
        [COMPLETE_WORLD_VARIABLES, [world_region]], names=[variable_level, region_level]
    )
    model_region_consistency_checking_variables = [
        v
        for v in COMPLETE_MODEL_REGION_VARIABLES
        # Avoid double counting with "Energy|Demand|Transportation"
        if SECTOR_DOMESTIC_AVIATION not in v
    ]
    model_region_consistency_checking = pd.MultiIndex.from_product(
        [model_region_consistency_checking_variables, model_regions],
        names=[variable_level, region_level],
    )

    res = world_internal_consistency_checking.append(model_region_consistency_checking)

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


def get_default_internal_conistency_checking_tolerances() -> (
    dict[str, dict[str, float]] | dict[str, dict[str, PINT_SCALAR]]
):
    """
    Get default tolerances used when checking the internal consistency of data

    Behaviour varies depending on whether [openscm_units][] is available or not.
    """
    try:
        import openscm_units

        Q = openscm_units.unit_registry.Quantity

        default_tolerances = {
            "Emissions|BC": dict(rtol=1e-3, atol=Q(1e-3, "Mt BC/yr")),
            "Emissions|CH4": dict(rtol=1e-3, atol=Q(1e-2, "Mt CH4/yr")),
            "Emissions|CO": dict(rtol=1e-3, atol=Q(1e-1, "Mt CO/yr")),
            "Emissions|CO2": dict(rtol=1e-3, atol=Q(1e0, "Mt CO2/yr")),
            "Emissions|NH3": dict(rtol=1e-3, atol=Q(1e-2, "Mt NH3/yr")),
            "Emissions|NOx": dict(rtol=1e-3, atol=Q(1e-2, "Mt NO2/yr")),
            "Emissions|OC": dict(rtol=1e-3, atol=Q(1e-3, "Mt OC/yr")),
            "Emissions|Sulfur": dict(rtol=1e-3, atol=Q(1e-2, "Mt SO2/yr")),
            "Emissions|VOC": dict(rtol=1e-3, atol=Q(1e-2, "Mt VOC/yr")),
            "Emissions|N2O": dict(rtol=1e-3, atol=Q(1e-1, "kt N2O/yr")),
        }

    except ImportError:
        default_tolerances = {
            "Emissions|BC": dict(rtol=1e-3, atol=1e-6),
            "Emissions|CH4": dict(rtol=1e-3, atol=1e-6),
            "Emissions|CO": dict(rtol=1e-3, atol=1e-6),
            "Emissions|CO2": dict(rtol=1e-3, atol=1e-6),
            "Emissions|NH3": dict(rtol=1e-3, atol=1e-6),
            "Emissions|NOx": dict(rtol=1e-3, atol=1e-6),
            "Emissions|OC": dict(rtol=1e-3, atol=1e-6),
            "Emissions|Sulfur": dict(rtol=1e-3, atol=1e-6),
            "Emissions|VOC": dict(rtol=1e-3, atol=1e-6),
            "Emissions|N2O": dict(rtol=1e-3, atol=1e-6),
        }

    return default_tolerances


def is_internally_consistent(
    df: pd.DataFrame,
    model_regions: tuple[str, ...],
    tolerances: dict[str, dict[str, float | PINT_SCALAR]],
    world_region: str = "World",
    region_level: str = "region",
    variable_level: str = "variable",
    unit_level: str = "unit",
) -> None:
    try:
        import pint
    except ImportError:
        pint = None

    internal_consistency_checking_index = get_internal_consistency_checking_index(
        model_regions=model_regions
    )

    # Hard-code the logic here
    # because that's what is needed for consistency with the rest of the module.
    # If you sum over the sectors
    # and regions of the index which provides internal consistency,
    # you should get the reported totals.
    def get_aggregate_variable(v: str) -> str:
        return "|".join(v.split("|")[:2])

    for variable, df_variable in df.groupby(
        df.index.get_level_values(variable_level).map(get_aggregate_variable)
    ):
        df_variable_reported_total = df_variable.loc[
            (df_variable.index.get_level_values(variable_level) == variable)
            & (df_variable.index.get_level_values(region_level) == world_region)
        ]
        if df_variable_reported_total.empty:
            # Nothing reported so can move on
            continue

        internal_consistency_checking_locator = multi_index_match(
            df_variable.index, internal_consistency_checking_index
        )

        # TODO: split out a function like
        # assert_reported_matches_sum_of_components
        # This will have to be extremely specific to this setup
        # to be able to handle the fact that "World"
        # isn't really a region but we have to report it this way
        # within the ScenarioMIP context.
        df_variable_aggregate = set_new_single_value_levels(
            combine_species(
                groupby_except(
                    split_sectors(
                        df_variable.loc[internal_consistency_checking_locator],
                        bottom_level="sectors",
                    ),
                    ["sectors", "region"],
                ).sum()
            ),
            {region_level: world_region},
        ).reorder_levels(df_variable.index.names)

        tolerances_variable = {}
        for k, v in tolerances[variable].items():
            if pint is not None and isinstance(v, pint.Quantity):
                if k == "atol":
                    variable_units = df_variable.index.get_level_values(
                        unit_level
                    ).unique()
                    if len(variable_units) > 1:
                        msg = (
                            "Cannot use pint conversion "
                            "if your data contains different units. "
                            f"For {variable=}, we have {variable_units=}"
                        )
                        raise ValueError(msg)

                    tolerances_variable[k] = v.to(variable_units[0]).m

                elif k == "rtol":
                    tolerances_variable[k] = v.to("dimensionless").m

                else:
                    raise NotImplementedError(k)

            else:
                tolerances_variable[k] = v

        comparison_variable = compare_close(
            left=df_variable_reported_total,
            right=df_variable_aggregate,
            left_name="reported_total",
            right_name="derived_from_input",
            **tolerances_variable,
        )

        if not comparison_variable.empty:
            # df_variable_not_used = df_variable.loc[
            #     ~internal_consistency_checking_locator
            # ]
            raise InternalConsistencyError(
                differences=comparison_variable,
                data_that_was_summed=df_variable,
                # data_that_was_not_summed=df_variable_not_used,
                tolerances=tolerances_variable,
            )


@define
class ToCompleteResult:
    """
    Result of calling `to_complete`
    """

    complete: pd.DataFrame
    """Complete [pd.DataFrame][pandas.DataFrame]"""

    assumed_zero: pd.DataFrame | None
    """
    The timeseries that were assumed to be zero to make `self.complete`

    If `None`, no timeseries were assumed to be zero.
    """


def to_complete(
    indf: pd.DataFrame,
    model_regions: tuple[str, ...],
    unit_level: str = "unit",
    variable_level: str = "variable",
    region_level: str = "region",
    world_region: str = "World",
) -> ToCompleteResult:
    assert_only_working_on_variable_unit_region_variations(indf)

    complete_index = get_complete_timeseries_index(
        model_regions=model_regions,
        region_level=region_level,
        variable_level=variable_level,
        world_region=world_region,
    )

    keep = multi_index_lookup(indf, complete_index)
    missing_indexes = get_missing_levels(
        keep.index, complete_index=complete_index, unit_col=unit_level
    )
    if missing_indexes.empty:
        res = ToCompleteResult(complete=keep, assumed_zero=None)
    else:
        keep_split = split_sectors(keep, middle_level="species")

        species_unit_map = {
            species: unit
            for species, unit in keep_split.index.droplevel(
                keep_split.index.names.difference(["species", unit_level])
            )
            .drop_duplicates()
            .reorder_levels(["species", unit_level])
        }
        missing_indexes_split = split_sectors(missing_indexes)
        zeros_index_split = create_levels_based_on_existing(
            missing_indexes_split, {unit_level: ("species", species_unit_map)}
        )
        zeros_index = combine_sectors(zeros_index_split, middle_level="species")

        other_levels_deduped = indf.index.droplevel(
            [variable_level, unit_level, region_level]
        ).drop_duplicates()
        if other_levels_deduped.shape[0] != 1:
            msg = f"Multiple values in other levels:\n{other_levels_deduped=}"
            raise AssertionError(msg)

        extra_levels = {
            level: value
            for level, value in zip(other_levels_deduped.names, other_levels_deduped[0])
        }
        assumed_zero = set_new_single_value_levels(
            pd.DataFrame(
                np.zeros((zeros_index.shape[0], keep.shape[1])),
                columns=keep.columns,
                index=zeros_index,
            ),
            extra_levels,
            copy=False,
        )
        complete = pd.concat([keep, assumed_zero.reorder_levels(keep.index.names)])
        res = ToCompleteResult(complete=complete, assumed_zero=assumed_zero)

    return res


def to_gridding_sectors(
    indf: pd.DataFrame, region_level: str = "region", world_region: str = "World"
) -> pd.DataFrame:
    # Processing is way easier if we split into two DataFrame's
    # and stack the sectors
    world_locator = indf.index.get_level_values(region_level) == world_region

    # Data that is at the world level i.e. has no region information
    sector_df = (
        split_sectors(
            indf.loc[world_locator].reset_index("region", drop=True),
            bottom_level="sectors",
        )
        .stack()
        .unstack("sectors")
    )
    # Data with region information
    region_sector_df = (
        split_sectors(indf.loc[~world_locator], bottom_level="sectors")
        .stack()
        .unstack("sectors")
    )

    # Move domestic aviation to the global level
    domestic_aviation_sum = groupby_except(
        region_sector_df[SECTOR_DOMESTIC_AVIATION], region_level
    ).sum()
    sector_df["Aircraft"] = (
        sector_df["Energy|Demand|Bunkers|International Aviation"]
        + domestic_aviation_sum
    )
    # The gridding transport sector excludes the aviation (which we just moved)
    region_sector_df["Energy|Demand|Transportation"] = (
        region_sector_df["Energy|Demand|Transportation"]
        - region_sector_df[SECTOR_DOMESTIC_AVIATION]
    )
    # Having done the move, drop the levels we no longer use
    sector_df = sector_df.drop(
        ["Energy|Demand|Bunkers|International Aviation"], axis="columns"
    )
    region_sector_df = region_sector_df.drop([SECTOR_DOMESTIC_AVIATION], axis="columns")

    # Now it's very straight-forward
    # Rename shipping at the world level without a loop
    # because this is the only change
    sector_df_gridding = sector_df.rename(
        {"Energy|Demand|Bunkers|International Shipping": "International Shipping"},
        axis="columns",
    )

    # Get the region-sector gridding df started
    region_sector_df_gridding = region_sector_df.rename(
        {"Energy|Demand|Bunkers|International Aviation": "Transportation Sector"},
        axis="columns",
    )
    # Do other compilations.
    # We can do this here with confidence
    # because we assume that the users have used `to_complete`
    # before calling this function.
    for gridding_sector, components in (
        ("Agriculture", ["AFOLU|Agriculture"]),
        (
            "Agricultural Waste Burning",
            [
                "AFOLU|Agricultural Waste Burning",
                # Hmmm, almost definitely wrong
                "AFOLU|Land|Harvested Wood Products",
                "AFOLU|Land|Land Use and Land-Use Change",
                "AFOLU|Land|Other",
                "AFOLU|Land|Wetlands",
            ],
        ),
        ("Energy Sector", ["Energy|Supply"]),
        ("Forest Burning", ["AFOLU|Land|Fires|Forest Burning"]),
        ("Grassland Burning", ["AFOLU|Land|Fires|Grassland Burning"]),
        (
            "Industrial Sector",
            [
                "Energy|Demand|Industry",
                "Industrial Processes",
                "Energy|Demand|Other Sector",
                "Other",
                "Other Capture and Removal",
            ],
        ),
        ("Peat Burning", ["AFOLU|Land|Fires|Peat Burning"]),
        (
            "Residential Commercial Other",
            ["Energy|Demand|Residential and Commercial and AFOFI"],
        ),
        ("Solvents Production and Application", ["Product Use"]),
        ("Waste", ["Waste"]),
    ):
        region_sector_df_gridding[gridding_sector] = region_sector_df_gridding[
            components
        ].sum(axis="columns")
        region_sector_df_gridding = region_sector_df_gridding.drop(
            list(set(components) - {gridding_sector}), axis="columns"
        )

    sector_df_gridding_like_input = combine_sectors(
        set_new_single_value_levels(
            sector_df_gridding.unstack().stack("sectors"), {region_level: world_region}
        ),
        bottom_level="sectors",
    )
    region_sector_df_gridding_like_input = combine_sectors(
        region_sector_df_gridding.unstack().stack("sectors"),
        bottom_level="sectors",
    )

    res = pd.concat(
        [
            df.reorder_levels(indf.index.names)
            for df in [
                sector_df_gridding_like_input,
                region_sector_df_gridding_like_input,
            ]
        ]
    )

    return res
