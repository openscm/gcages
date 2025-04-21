"""
Pre-processing part of the workflow

This is extremely fiddly because the data is inhomogenous
(some is reported at the regional level,
other bits only at the global level
and we need to be able to move between the two).

This module implements the following logic.
There are a lot of global variables.
It is likely possible to split this out.
However, it would be extremely difficult to test
that the individual components can be altered
and the whole stays consistent.
As a result, we have written it like this to make clearer
that this entire module is more or less coupled,
If you alter any of the global variables,
we don't guarantee correct behaviour.

The underlying logic is this:

- we're doing region-sector harmonisation
- hence we need regions and sectors lined up very specifically with CEDS

If you want simpler pre-processing, use one of the other pre-processing tools.
"""

from __future__ import annotations

import itertools
import multiprocessing
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from attrs import asdict, define
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.indexing import multi_index_lookup
from pandas_openscm.parallelisation import ParallelOpConfig, apply_op_parallel_progress

from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_index_levels,
    assert_index_is_multiindex,
    assert_only_working_on_variable_unit_region_variations,
)
from gcages.completeness import assert_all_groups_are_complete
from gcages.exceptions import MissingOptionalDependencyError
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import assert_frame_equal, compare_close
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

REQUIRED_GRIDDING_SPECIES_IAMC: tuple[str, ...] = (
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
Species to prepare for gridding (in the IAMC naming convention)

All species in this tuple must be provided in the input.

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

INDUSTRIAL_SECTOR_CEDS_COMPONENTS_IAMC: tuple[str, ...] = (
    "Energy|Demand|Industry",
    "Energy|Demand|Other Sector",
    "Industrial Processes",
    "Other",
)
"""
Sectors (in the IAMC naming convention) that make up the CEDS industrial sector

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

INDUSTRIAL_SECTOR_CEDS: str = "Industrial Sector"
"""
Sector name used for the industrial sector in CEDS
"""

DOMESTIC_AVIATION_SECTOR_IAMC: str = "Energy|Demand|Transportation|Domestic Aviation"
"""
Assumed sector for the domestic aviation sector (IAMC naming convention)
"""

INTERNATIONAL_AVIATION_SECTOR_IAMC: str = "Energy|Demand|Bunkers|International Aviation"
"""
Assumed reporting for the international aviation sector (IAMC naming convention)
"""

AVIATION_SECTOR_CEDS: str = "Aircraft"
"""
Sector name used for aviation in CEDS
"""

TRANSPORTATION_SECTOR_IAMC: str = "Energy|Demand|Transportation"
"""
Assumed reporting for the transportation sector (IAMC naming convention)
"""

TRANSPORTATION_SECTOR_CEDS: str = "Transportation Sector"
"""
Sector name used for transport in CEDS
"""

AGRICULTURE_SECTOR_COMPONENTS_IAMC: tuple[str, ...] = (
    "AFOLU|Agriculture",
    "AFOLU|Land|Land Use and Land-Use Change",
    "AFOLU|Land|Harvested Wood Products",
    "AFOLU|Land|Other",
    "AFOLU|Land|Wetlands",
)
"""
Sectors that make up the CEDS agriculture sector (IAMC naming convention)

Not all of these components are required.
If they are not supplied, they are assumed to be zero.

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

AGRICULTURE_SECTOR_CEDS: str = "Agriculture"
"""
Sector name used for the agriculture sector in CEDS
"""

REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC: tuple[str, ...] = (
    "Energy|Supply",
    *INDUSTRIAL_SECTOR_CEDS_COMPONENTS_IAMC,
    "Energy|Demand|Residential and Commercial and AFOFI",
    "Product Use",
    # Technically, domestic aviation could be reported just
    # at the world level and it would be fine.
    # In practice, no-one does that and the logic is much simpler
    # if we assume it has to be reported regionally
    # (because then domestic aviation and transport are on the same regional 'grid')
    # so do that for now.
    DOMESTIC_AVIATION_SECTOR_IAMC,
    TRANSPORTATION_SECTOR_IAMC,
    # The rest of TRANSPORTATION_SECTOR_REQUIRED_REPORTING_IAMC
    # can be reported at the world level only and it is fine
    "Waste",
    # Note: AFOLU|Agriculture is the only compulsory component for agriculture
    # which is why we don't use
    # *AGRICULTURE_SECTOR_COMPONENTS_IAMC
    "AFOLU|Agriculture",
    "AFOLU|Agricultural Waste Burning",
    "AFOLU|Land|Fires|Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning",
    "AFOLU|Land|Fires|Peat Burning",
)
"""
Sectors (IAMC naming) required in the input at the regional level for gridding

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

REQUIRED_GRIDDING_SECTORS_WORLD_IAMC: tuple[str, ...] = (
    INTERNATIONAL_AVIATION_SECTOR_IAMC,
    "Energy|Demand|Bunkers|International Shipping",
)
"""
Sectors (IAMC naming) required in the input at the world level for gridding

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

REQUIRED_REGIONAL_VARIABLES_IAMC: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_IAMC
    for sector in REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC
)
"""
Variables required at the regional level (IAMC naming convention)

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

REQUIRED_WORLD_VARIABLES_IAMC: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_IAMC
    for sector in REQUIRED_GRIDDING_SECTORS_WORLD_IAMC
)
"""
Variables required at the world level (IAMC naming convention)

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

REQUIRED_REGIONAL_INDEX_IAMC = pd.MultiIndex(
    names=["variable"],
    levels=[REQUIRED_REGIONAL_VARIABLES_IAMC],
    codes=[np.arange(len(REQUIRED_REGIONAL_VARIABLES_IAMC))],
)
"""
The required index for regional data for gridding (IAMC naming convention)

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

REQUIRED_REGIONAL_VARIABLES_IAMC: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_IAMC
    for sector in REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC
)
"""
Variables required at the regional level (IAMC naming convention)

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

ALL_RELEVANT_REGIONAL_VARIABLES_IAMC = tuple(
    # Use of set avoids double counting
    {
        *REQUIRED_REGIONAL_VARIABLES_IAMC,
        *(
            f"Emissions|{species}|{sector}"
            for species in REQUIRED_GRIDDING_SPECIES_IAMC
            for sector in AGRICULTURE_SECTOR_COMPONENTS_IAMC
        ),
    }
)
"""
Variables that are relevant at the regional level (IAMC naming convention)

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

ALL_RELEVANT_REGIONAL_INDEX_IAMC = pd.MultiIndex(
    names=["variable"],
    levels=[ALL_RELEVANT_REGIONAL_VARIABLES_IAMC],
    codes=[np.arange(len(ALL_RELEVANT_REGIONAL_VARIABLES_IAMC))],
)

"""
The index of all relevant levels for regional data for gridding (IAMC naming convention)

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

REQUIRED_WORLD_INDEX_IAMC = pd.MultiIndex(
    names=["variable"],
    levels=[REQUIRED_WORLD_VARIABLES_IAMC],
    codes=[np.arange(len(REQUIRED_WORLD_VARIABLES_IAMC))],
)
"""
The required index for world data for gridding (IAMC naming convention)

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

ALL_RELEVANT_WORLD_VARIABLES_IAMC = REQUIRED_WORLD_VARIABLES_IAMC
"""
Variables that are relevant at the world level (IAMC naming convention)

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

ALL_RELEVANT_WORLD_INDEX_IAMC = REQUIRED_WORLD_INDEX_IAMC
"""
The index of all relevant levels for world data for gridding (IAMC naming convention)

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

REAGGREGATED_TO_GRIDDING_SECTOR_MAP: dict[str, str] = {
    "Energy|Supply": "Energy Sector",
    INDUSTRIAL_SECTOR_CEDS: INDUSTRIAL_SECTOR_CEDS,
    "Energy|Demand|Residential and Commercial and AFOFI": "Residential Commercial Other",  # noqa: E501
    "Product Use": "Solvents Production and Application",
    TRANSPORTATION_SECTOR_CEDS: TRANSPORTATION_SECTOR_CEDS,
    "Waste": "Waste",
    AVIATION_SECTOR_CEDS: AVIATION_SECTOR_CEDS,
    "Energy|Demand|Bunkers|International Shipping": "International Shipping",
    AGRICULTURE_SECTOR_CEDS: AGRICULTURE_SECTOR_CEDS,
    "AFOLU|Agricultural Waste Burning": "Agricultural Waste Burning",
    "AFOLU|Land|Fires|Forest Burning": "Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning": "Grassland Burning",
    "AFOLU|Land|Fires|Peat Burning": "Peat Burning",
}
"""
Map from re-aggreated variables to sectors used for gridding
"""

REQUIRED_WORLD_SECTORS_CEDS: tuple[str, ...] = ("Aircraft", "International Shipping")
"""
Sectors that are required at the world level (CEDS naming)

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""

REQUIRED_REGIONAL_SECTORS_CEDS: tuple[str, ...] = (
    "Energy Sector",
    INDUSTRIAL_SECTOR_CEDS,
    "Solvents Production and Application",
    TRANSPORTATION_SECTOR_CEDS,
    "Waste",
    AGRICULTURE_SECTOR_CEDS,
    "Agricultural Waste Burning",
    "Forest Burning",
    "Grassland Burning",
    "Peat Burning",
)
"""
Sectors that are required at the regional level (CEDS naming)

This variable is provided as a global variable for clarity and consistency.
If you change it, we do not guarantee the package's performance.
"""


CO2_FOSSIL_SECTORS_CEDS: tuple[str, ...] = (
    "Energy Sector",
    INDUSTRIAL_SECTOR_CEDS,
    "Residential Commercial Other",
    "Solvents Production and Application",
    TRANSPORTATION_SECTOR_CEDS,
    "Waste",
    AVIATION_SECTOR_CEDS,
    "International Shipping",
)
"""
Sectors (CEDS naming) whose CO2 emissions originate from fossil reservoirs
"""

CO2_BIOSPHERE_SECTORS_CEDS: tuple[str, ...] = (
    AGRICULTURE_SECTOR_CEDS,
    "Agricultural Waste Burning",
    "Forest Burning",
    "Grassland Burning",
    "Peat Burning",
)
"""
Sectors (CEDS naming) whose CO2 emissions originate from the biosphere (land pool)
"""


@define
class SplitData:
    """
    Container for holding data split into regional and world data
    """

    world: pd.DataFrame
    """Data at the world level"""

    regional: pd.DataFrame
    """Data at the regional level"""


def split_world_and_regional_data(indf: pd.DataFrame, world_region: str) -> SplitData:
    world_locator = indf.index.get_level_values("region") == world_region
    world_data = indf.loc[world_locator]
    regional_data = indf.loc[~world_locator]

    return SplitData(world=world_data, regional=regional_data)


def assert_no_obvious_double_counting(sum_over_values: list[str]) -> None:
    double_counting = [
        s
        for s in sum_over_values
        # If the value is a subset of another value,
        # we will double count
        if any(f"{s}|" in s_other for s_other in sum_over_values)
    ]
    if double_counting:
        msg = f"Likely double counting {double_counting=} {sum_over_values=}"

        raise AssertionError(msg)


def get_region_sector_totals(
    indf: pd.DataFrame, world_region: str, region_level: str
) -> pd.DataFrame:
    try:
        from pandas_indexing.core import assignlevel, extractlevel, formatlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_region_sector_totals", requirement="pandas_indexing"
        ) from exc

    indf_split = extractlevel(indf, variable="{table}|{species}|{sectors}")

    assert_no_obvious_double_counting(
        indf_split.index.get_level_values("sectors").unique().tolist()
    )
    assert_no_obvious_double_counting(
        indf_split.index.get_level_values("region").unique().tolist()
    )

    res = formatlevel(
        assignlevel(
            indf_split.groupby(
                indf_split.index.names.difference([region_level, "sectors"])
            ).sum(),
            region=world_region,
        ),
        variable="{table}|{species}",
        drop=True,
    )

    return res


def get_raw_data_for_gridding(indf: pd.DataFrame, world_region: str) -> pd.DataFrame:
    # hi
    pass
    """
    You'll likely want to call

    [assert_data_is_compatible_with_pre_processing][(m).]
    before using this as it gives better error messages if you have missing data.
    """
    split_data = split_world_and_regional_data(indf, world_region=world_region)

    regional_data = multi_index_lookup(
        split_data.regional, ALL_RELEVANT_REGIONAL_INDEX_IAMC
    )
    world_data = multi_index_lookup(split_data.world, ALL_RELEVANT_WORLD_INDEX_IAMC)

    res = pd.concat([world_data, regional_data.reorder_levels(world_data.index.names)])

    return res


def drop_out_domestic_aviation(indf: pd.DataFrame) -> pd.DataFrame:
    if TRANSPORTATION_SECTOR_IAMC not in DOMESTIC_AVIATION_SECTOR_IAMC:
        msg = "Logic of this is broken"
        raise AssertionError(msg)

    res = indf.loc[
        ~indf.index.get_level_values("variable").str.endswith(
            DOMESTIC_AVIATION_SECTOR_IAMC
        )
    ]

    return res


class InternalConsistencyError(ValueError):
    """
    Raised when there is an internal consistency issue in the data

    Specifically, the sum of components doesn't match some total
    """

    def __init__(
        self,
        differences: pd.DataFrame,
        data_that_was_summed: pd.DataFrame,
    ) -> None:
        differences_variables = differences.index.get_level_values("variable").unique()
        data_that_was_summed_relevant_for_differences = data_that_was_summed[
            data_that_was_summed.index.get_level_values("variable").map(
                lambda x: any(v in x for v in differences_variables)
            )
        ].index.to_frame(index=False)

        error_msg = (
            "Summing the components does not equal the total. "
            f"Differences:\n{differences}\n"
            "This is the data we used in the sum:\n"
            f"{data_that_was_summed_relevant_for_differences}"
        )

        super().__init__(error_msg)


def assert_data_is_compatible_with_pre_processing(
    indf: pd.DataFrame,
    world_region: str,
    region_level: str,
    rtol_internal_consistency: float,
    atol_internal_consistency: float,
) -> None:
    indf_drop_all_nan_cols = indf.dropna(how="all", axis="columns")
    if indf_drop_all_nan_cols.isnull().any().any():
        msg = f"NaNs after dropping unreported times:\n{indf_drop_all_nan_cols}"
        raise AssertionError(msg)

    assert_only_working_on_variable_unit_region_variations(indf)

    split_data = split_world_and_regional_data(indf, world_region=world_region)

    assert_all_groups_are_complete(
        split_data.world, complete_index=REQUIRED_WORLD_INDEX_IAMC
    )
    assert_all_groups_are_complete(
        split_data.regional, complete_index=REQUIRED_REGIONAL_INDEX_IAMC
    )

    data_for_gridding_raw = get_raw_data_for_gridding(indf, world_region=world_region)

    # Drop out domestic aviation because it is a subsector of transport
    data_for_gridding_raw_to_sum = drop_out_domestic_aviation(data_for_gridding_raw)

    data_for_gridding_totals = get_region_sector_totals(
        data_for_gridding_raw_to_sum,
        world_region=world_region,
        region_level=region_level,
    )

    indf_global_totals = multi_index_lookup(
        split_data.world, data_for_gridding_totals.index
    )

    differences = compare_close(
        data_for_gridding_totals,
        indf_global_totals,
        left_name="total_from_sectors_and_regions_used_in_gridding",
        right_name="total_reported_in_input",
        rtol=rtol_internal_consistency,
        atol=atol_internal_consistency,
    )
    if not differences.empty:
        raise InternalConsistencyError(
            differences=differences, data_that_was_summed=data_for_gridding_raw_to_sum
        )


def assert_data_has_required_internal_consistency(  # noqa: PLR0913
    indf: pd.DataFrame,
    world_region: str,
    region_level: str,
    time_name: str,
    rtol_internal_consistency: float,
    atol_internal_consistency: float,
) -> None:
    split_data = split_world_and_regional_data(indf, world_region=world_region)

    # Only check gridding data.
    # Other data will just be taken from World, so not our problem
    # if it's inconsistent.
    gridding_data_world = split_data.world.loc[
        split_data.world.index.get_level_values("variable").isin(
            REQUIRED_WORLD_VARIABLES_IAMC
        )
    ]

    # Want to sum over everything except domestic aviation
    # to not double count with transport.
    regional_variables_to_include_in_sum = [
        v
        for v in ALL_RELEVANT_REGIONAL_VARIABLES_IAMC
        if DOMESTIC_AVIATION_SECTOR_IAMC not in v
    ]
    gridding_data_regional = split_data.regional.loc[
        split_data.regional.index.get_level_values("variable").isin(
            regional_variables_to_include_in_sum
        )
    ]

    gridding_data_world_region_sector_sum = get_region_sector_totals(
        gridding_data_world, world_region=world_region, region_level=region_level
    )
    gridding_data_regional_region_sector_sum = get_region_sector_totals(
        gridding_data_regional,
        world_region=world_region,
        region_level=region_level,
    )

    gridding_data_region_sector_sum = (
        gridding_data_world_region_sector_sum + gridding_data_regional_region_sector_sum
    )

    reported_region_sector_sum = multi_index_lookup(
        indf, gridding_data_regional_region_sector_sum.index
    )

    assert_frame_equal(
        gridding_data_region_sector_sum,
        reported_region_sector_sum,
        rtol=rtol_internal_consistency,
        atol=atol_internal_consistency,
    )


@define
class CMIP7ScenarioMIPPreProcessingResult:
    """
    Result of pre-processing with [CMIP7ScenarioMIPPreProcessor][]

    This has more components than normal,
    because we need to support both the 'normal' global path
    and harmonising at the region-sector level.
    """

    gridding_workflow_emissions: pd.DataFrame
    """
    Emissions that can be used with the gridding workflow
    """

    global_workflow_emissions: pd.DataFrame
    """
    Emissions that can be used with the 'normal' global workflow
    """

    global_workflow_emissions_raw_names: pd.DataFrame
    """
    Emissions consistent with those that can be used with the 'normal' global workflow

    The difference is that these are reported with CMIP7 ScenarioMIP naming,
    which isn't compatible with our SCM runners (for example),
    so is probably not what you want to use,
    but perhaps helpful for plotting and direct comparisons.
    """


def unstack_sector(indf: pd.DataFrame, time_name: str) -> pd.DataFrame:
    try:
        from pandas_indexing.core import extractlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "unstack_sector", requirement="pandas_indexing"
        ) from exc

    res = (
        extractlevel(indf, variable="{table}|{species}|{sectors}")
        .unstack("sectors")
        .stack(time_name, future_stack=True)
    )

    return res


def stack_sector_and_return_to_variable(
    indf: pd.DataFrame, time_name: str, sectors_name: str
) -> pd.DataFrame:
    try:
        from pandas_indexing.core import formatlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "stack_sector_and_return_to_variable", requirement="pandas_indexing"
        ) from exc

    res = formatlevel(
        indf.unstack(time_name).stack(sectors_name, future_stack=True),
        variable="{table}|{species}|{sectors}",
        drop=True,
    )

    return res


def reclassify_aviation_emissions(
    indf: pd.DataFrame, world_region: str, time_name: str, region_level: str
) -> pd.DataFrame:
    split_data = split_world_and_regional_data(indf, world_region=world_region)

    regional_sector_cols = unstack_sector(split_data.regional, time_name=time_name)
    world_sector_cols = unstack_sector(split_data.world, time_name=time_name)

    domestic_aviation_sum = (
        regional_sector_cols[DOMESTIC_AVIATION_SECTOR_IAMC]
        .groupby(regional_sector_cols.index.names.difference([region_level]))
        .sum()
    )
    world_sector_cols[AVIATION_SECTOR_CEDS] = (
        world_sector_cols[INTERNATIONAL_AVIATION_SECTOR_IAMC] + domestic_aviation_sum
    )

    regional_sector_cols[TRANSPORTATION_SECTOR_CEDS] = (
        regional_sector_cols[TRANSPORTATION_SECTOR_IAMC]
        - regional_sector_cols[DOMESTIC_AVIATION_SECTOR_IAMC]
    )

    # Drop out sectors we're not using
    world_sector_cols = world_sector_cols[
        world_sector_cols.columns.difference([INTERNATIONAL_AVIATION_SECTOR_IAMC])
    ]
    regional_sector_cols = regional_sector_cols[
        regional_sector_cols.columns.difference(
            [DOMESTIC_AVIATION_SECTOR_IAMC, TRANSPORTATION_SECTOR_IAMC]
        )
    ]

    get_out = partial(
        stack_sector_and_return_to_variable,
        sectors_name="sectors",
        time_name=time_name,
    )
    world_out = get_out(world_sector_cols)
    regional_out = get_out(regional_sector_cols)

    res = pd.concat([world_out, regional_out.reorder_levels(world_out.index.names)])

    return res


def aggregate_sector(
    indf: pd.DataFrame,
    sector_out: str,
    sector_components: list[str],
    time_name: str,
    allow_missing: list[str] | None = None,
) -> pd.DataFrame:
    sector_cols = unstack_sector(indf, time_name=time_name)

    to_sum = sector_components
    if allow_missing is not None:
        missing = {c for c in to_sum if c not in sector_cols}
        # Anything which is missing and allowed to be missing,
        # we can drop from to_sum
        to_drop_from_sum = missing.intersection(set(allow_missing))
        to_sum = list(set(to_sum) - to_drop_from_sum)

    sector_cols[sector_out] = sector_cols[to_sum].sum(
        axis="columns", min_count=len(to_sum)
    )
    sector_cols = sector_cols.drop(to_sum, axis="columns")

    res = stack_sector_and_return_to_variable(
        sector_cols, time_name=time_name, sectors_name="sectors"
    )

    return res


def rename_and_filter_to_ceds_aligned_sectors(
    indf: pd.DataFrame, time_name: str
) -> pd.DataFrame:
    sector_cols = unstack_sector(indf, time_name=time_name)

    renamed = sector_cols.rename(
        REAGGREGATED_TO_GRIDDING_SECTOR_MAP, axis="columns", errors="raise"
    )

    res = stack_sector_and_return_to_variable(
        renamed, time_name=time_name, sectors_name="sectors"
    )
    # Funny things can happen when unstacking for some reason
    res = res.dropna(how="all")

    return res


def aggregate_gridding_workflow_emissions_to_global_workflow_emissions(  # noqa: PLR0913
    gridding_emissions: pd.DataFrame,
    world_region: str,
    time_name: str,
    region_level: str,
    global_workflow_co2_fossil_sector_iamc: str,
    global_workflow_co2_biosphere_sector_iamc: str,
    co2_fossil_sectors_ceds: tuple[str, ...] = CO2_FOSSIL_SECTORS_CEDS,
    co2_biosphere_sectors_ceds: tuple[str, ...] = CO2_BIOSPHERE_SECTORS_CEDS,
) -> pd.DataFrame:
    """
    Aggregate the gridding workflow emissions to global workflow emissions

    Parameters
    ----------
    gridding_emissions
        [pd.DataFrame][pandas.DataFrame] used for gridding

    world_region
        Name of the world/world total region

    time_name
        Name of the columns (i.e. time axis) in `gridding_emissions`

    region_level
        The name of the index level which contains region-level data

    global_workflow_co2_fossil_sector_iamc
        Name of the global workflow CO2 fossil sector using IAMC naming conventions

    global_workflow_co2_biosphere_sector_iamc
        Name of the global workflow CO2 biosphere sector using IAMC naming conventions

    co2_fossil_sectors_ceds
        CEDS CO2 sectors that should be included in the fossil CO2 sector

    co2_biosphere_sectors_ceds
        CEDS CO2 sectors that should be included in the biosphere CO2 sector

    Returns
    -------
    :
        Derived output that can be used with the global workflow
    """
    try:
        from pandas_indexing.core import assignlevel, formatlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "stack_sector_and_return_to_variable", requirement="pandas_indexing"
        ) from exc

    split_data = split_world_and_regional_data(
        gridding_emissions, world_region=world_region
    )
    sector_cols_world = unstack_sector(split_data.world, time_name=time_name)
    sector_cols_regional = unstack_sector(split_data.regional, time_name=time_name)

    in_both_world_and_regional = sector_cols_world.columns.intersection(
        sector_cols_regional.columns
    )
    if not in_both_world_and_regional.empty:
        msg = (
            "The following sectors are in both world and regional data: "
            f"{in_both_world_and_regional}"
        )
        raise AssertionError(msg)

    missing = {*co2_biosphere_sectors_ceds, *co2_fossil_sectors_ceds}
    in_data = {*sector_cols_world.columns, *sector_cols_regional.columns}
    not_handled = in_data - missing
    if not_handled:
        msg = (
            "The following sectors "
            f"will not be included in the CO2 output: {not_handled}"
        )
        raise AssertionError(msg)

    missing = missing - in_data
    if missing:
        msg = f"The following sectors are missing from the expected sectors: {missing}"
        raise AssertionError(msg)

    def filter(idf: pd.DataFrame, co2: bool) -> pd.DataFrame:
        co2_locator = idf.index.get_level_values("species") == "CO2"
        if co2:
            return idf.loc[co2_locator]

        return idf.loc[~co2_locator]

    def get_out(idf: pd.DataFrame) -> pd.DataFrame:
        return formatlevel(
            idf.unstack(time_name), variable="{table}|{species}", drop=True
        )

    sector_cols_world_sum = sector_cols_world.sum(axis="columns", skipna=False)
    sector_cols_regional_s = sector_cols_regional.stack()
    sector_cols_regional_sum = sector_cols_regional_s.groupby(
        sector_cols_regional_s.index.names.difference([region_level, "sectors"])
    ).sum()
    non_co2 = get_out(
        filter(
            sector_cols_world_sum + sector_cols_regional_sum,
            co2=False,
        )
    )

    co2_world = filter(sector_cols_world, co2=True)
    co2_regional = filter(sector_cols_regional, co2=True)
    co2_fossil_sectors_from_world = [
        c for c in co2_world if c in co2_fossil_sectors_ceds
    ]
    co2_fossil_sectors_from_regional = list(
        set(co2_fossil_sectors_ceds) - set(co2_fossil_sectors_from_world)
    )

    co2_fossil_world_sum = co2_world[co2_fossil_sectors_from_world].sum(
        axis="columns", skipna=False
    )
    co2_fossil_regional_sum = (
        co2_regional[co2_fossil_sectors_from_regional]
        .stack()
        .groupby(co2_regional.index.names.difference([region_level, "sectors"]))
        .sum()
    )

    def get_out_co2(indf: pd.DataFrame, sector: str) -> pd.DataFrame:
        tmp = indf.to_frame(sector)
        tmp.columns.name = "sectors"

        return stack_sector_and_return_to_variable(
            tmp,
            time_name=time_name,
            sectors_name="sectors",
        )

    co2_fossil = get_out_co2(
        co2_fossil_world_sum + co2_fossil_regional_sum,
        sector=global_workflow_co2_fossil_sector_iamc,
    )
    co2_biosphere = get_out_co2(
        assignlevel(
            co2_regional[list(co2_biosphere_sectors_ceds)]
            .stack()
            .groupby(co2_regional.index.names.difference([region_level, "sectors"]))
            .sum(),
            region=world_region,
        ),
        sector=global_workflow_co2_biosphere_sector_iamc,
    )

    res = pd.concat(
        [
            co2_fossil,
            co2_biosphere.reorder_levels(co2_fossil.index.names),
            non_co2.reorder_levels(co2_fossil.index.names),
        ]
    ).sort_index(axis="columns")

    return res


def get_global_workflow_emissions_not_from_gridding_variables(  # noqa: PLR0913
    df_clean_units: pd.DataFrame,
    gridding_variables: list[str],
    level_separator: str,
    region_level: str,
    world_region: str,
    variable_level: str,
    unit_level: str,
) -> pd.DataFrame:
    species_from_gridding = {v.split(level_separator)[1] for v in gridding_variables}

    indf_variables = df_clean_units.index.get_level_values(variable_level).unique()
    not_used_in_gridding_variables = [
        v for v in indf_variables if not any(s in v for s in species_from_gridding)
    ]

    to_keep = df_clean_units.loc[
        df_clean_units.index.get_level_values(variable_level).isin(
            not_used_in_gridding_variables
        )
        & (df_clean_units.index.get_level_values(region_level) == world_region)
        # equiv not usable for now
        & ~df_clean_units.index.get_level_values(unit_level).str.contains("equiv")
    ]

    return to_keep


def do_pre_processing(  # noqa: PLR0913
    indf: pd.DataFrame,
    world_region: str,
    time_name: str,
    region_level: str,
    variable_level: str,
    unit_level: str,
    level_separator: str,
) -> CMIP7ScenarioMIPPreProcessingResult:
    assert_only_working_on_variable_unit_region_variations(indf)

    indf_reported_times = indf.dropna(how="all", axis="columns")
    indf_clean_units = strip_pint_incompatible_characters_from_units(
        indf_reported_times,
        units_index_level="unit",
    )
    data_for_gridding = get_raw_data_for_gridding(
        indf_clean_units, world_region=world_region
    )

    # From here on in, we are carrying both global and regional data,
    # but only at the regional detail we need.
    aviation_reclassified = reclassify_aviation_emissions(
        data_for_gridding,
        time_name=time_name,
        world_region=world_region,
        region_level=region_level,
    )

    industry_aggregated = aggregate_sector(
        aviation_reclassified,
        sector_out=INDUSTRIAL_SECTOR_CEDS,
        sector_components=list(INDUSTRIAL_SECTOR_CEDS_COMPONENTS_IAMC),
        time_name=time_name,
    )

    agriculture_aggregated = aggregate_sector(
        industry_aggregated,
        sector_out=AGRICULTURE_SECTOR_CEDS,
        sector_components=list(AGRICULTURE_SECTOR_COMPONENTS_IAMC),
        allow_missing=[
            v
            for v in AGRICULTURE_SECTOR_COMPONENTS_IAMC
            if v not in REQUIRED_GRIDDING_SECTORS_REGIONAL_IAMC
        ],
        time_name=time_name,
    )

    gridding_workflow_emissions = rename_and_filter_to_ceds_aligned_sectors(
        agriculture_aggregated, time_name=time_name
    )

    global_workflow_emissions_from_gridding_emissions = (
        aggregate_gridding_workflow_emissions_to_global_workflow_emissions(
            gridding_workflow_emissions,
            world_region=world_region,
            time_name=time_name,
            region_level=region_level,
            global_workflow_co2_fossil_sector_iamc="Energy and Industrial Processes",
            global_workflow_co2_biosphere_sector_iamc="AFOLU",
        )
    )

    global_workflow_emissions_not_from_gridding_emissions = (
        get_global_workflow_emissions_not_from_gridding_variables(
            df_clean_units=indf_clean_units,
            gridding_variables=gridding_workflow_emissions.index.get_level_values(
                variable_level
            )
            .unique()
            .tolist(),
            level_separator=level_separator,
            region_level=region_level,
            world_region=world_region,
            variable_level=variable_level,
            unit_level=unit_level,
        )
    )

    global_workflow_emissions_raw_names = pd.concat(
        [
            global_workflow_emissions_from_gridding_emissions,
            global_workflow_emissions_not_from_gridding_emissions.reorder_levels(
                global_workflow_emissions_from_gridding_emissions.index.names
            ),
        ]
    )

    global_workflow_emissions = update_index_levels_func(
        global_workflow_emissions_raw_names,
        {
            "variable": partial(
                convert_variable_name,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
    )

    res = CMIP7ScenarioMIPPreProcessingResult(
        gridding_workflow_emissions=gridding_workflow_emissions,
        global_workflow_emissions=global_workflow_emissions,
        global_workflow_emissions_raw_names=global_workflow_emissions_raw_names,
    )

    return res


def assert_no_nans_in_res_ms(res_ms: CMIP7ScenarioMIPPreProcessingResult) -> None:
    for attr in [
        "global_workflow_emissions",
        "gridding_workflow_emissions",
    ]:
        if getattr(res_ms, attr).isnull().any().any():
            ms = (
                res_ms.index.droplevels(
                    res_ms.index.names.difference(["model", "scenario"])
                )
                .drop_duplicates()
                .tolist()
            )
            msg = f"NaNs in res.{attr} for {ms}"
            raise AssertionError(msg)


def assert_column_type_unchanged_in_res_ms(
    res_ms: CMIP7ScenarioMIPPreProcessingResult, in_column_type: Any
) -> None:
    for attr in [
        "global_workflow_emissions",
        "global_workflow_emissions_raw_names",
        "gridding_workflow_emissions",
    ]:
        df = getattr(res_ms, attr)
        if df.columns.dtype != in_column_type:
            msg = "Column type has changed: " f"{df.columns.dtype=} {in_column_type}"
            raise AssertionError(msg)


def get_required_ceds_index(
    world_region: str, model_regions: Iterable[str]
) -> pd.MultiIndex:
    required_ceds_index = pd.MultiIndex.from_tuples(
        [
            *(
                (f"Emissions|{species}|{sector}", world_region)
                for species, sector in itertools.product(
                    REQUIRED_GRIDDING_SPECIES_IAMC, REQUIRED_WORLD_SECTORS_CEDS
                )
            ),
            *(
                (f"Emissions|{species}|{sector}", model_region)
                for species, sector, model_region in itertools.product(
                    REQUIRED_GRIDDING_SPECIES_IAMC,
                    REQUIRED_REGIONAL_SECTORS_CEDS,
                    model_regions,
                )
            ),
        ],
        names=["variable", "region"],
    )

    return required_ceds_index


def get_gridded_emissions_sectoral_regional_sum(
    indf: pd.DataFrame, time_name: str, region_level: str, world_region: str
) -> pd.DataFrame:
    try:
        from pandas_indexing.core import formatlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "stack_sector_and_return_to_variable", requirement="pandas_indexing"
        ) from exc

    sector_cols = unstack_sector(indf, time_name=time_name).stack()
    region_sector_sum = sector_cols.groupby(
        sector_cols.index.names.difference([region_level, "sectors"])
    ).sum()

    tmp = region_sector_sum.to_frame(world_region)
    tmp.columns.name = region_level

    res = formatlevel(
        tmp.unstack(time_name).stack(region_level, future_stack=True),
        variable="{table}|{species}",
        drop=True,
    ).sort_index(axis="columns")

    return res


@define
class CMIP7ScenarioMIPPreProcessor:
    """
    Pre-processor for CMIP7's ScenarioMIP

    Like most pre-processing,
    this is quite locked up and context-specific.
    The logic also goes through multiple layers,
    which makes it hard to clearly identify
    how techniques like dependency injection could actually be used.

    For more details of the logic, see [gcages.cmip7_scenariomip][].
    """

    rtol_internal_consistency: float = 1e-4
    """
    Relative tolerance to apply when checking the internal consistency of the data

    For example, when making sure that the sum of regional and sectoral information
    matches repoted totals.
    """

    atol_internal_consistency: float = 1e-6
    """
    Absolute tolerance to apply when checking the internal consistency of the data

    For example, when making sure that the sum of regional and sectoral information
    matches repoted totals.
    """

    world_region: str = "World"
    """
    String that identifies the world (i.e. global total) region
    """

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    progress: bool = True
    """
    Should progress bars be shown for each operation?
    """

    n_processes: int | None = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to `None` to process in serial.
    """

    def __call__(
        self, in_emissions: pd.DataFrame
    ) -> CMIP7ScenarioMIPPreProcessingResult:
        """
        Pre-process

        Parameters
        ----------
        in_emissions
            Emissions to pre-process

        Returns
        -------
        :
            Pre-processed emissions
        """
        if self.run_checks:
            assert_index_is_multiindex(in_emissions)
            assert_data_is_all_numeric(in_emissions)
            assert_has_index_levels(
                in_emissions, ["variable", "unit", "model", "scenario", "region"]
            )

            if in_emissions.columns.name != "year":
                msg = "The input emissions' column name should be 'year'"
                raise AssertionError(msg)

            for _, msdf in in_emissions.groupby(["model", "scenario"]):
                msdf_drop_all_nan_times = msdf.dropna(how="all", axis="columns")
                assert_data_is_compatible_with_pre_processing(
                    msdf_drop_all_nan_times,
                    world_region=self.world_region,
                    region_level="region",
                    rtol_internal_consistency=self.rtol_internal_consistency,
                    atol_internal_consistency=self.atol_internal_consistency,
                )

                assert_data_has_required_internal_consistency(
                    msdf_drop_all_nan_times,
                    world_region=self.world_region,
                    region_level="region",
                    time_name="year",
                    rtol_internal_consistency=self.rtol_internal_consistency,
                    atol_internal_consistency=self.atol_internal_consistency,
                )

        res_g = apply_op_parallel_progress(
            func_to_call=do_pre_processing,
            world_region=self.world_region,
            time_name="year",
            region_level="region",
            variable_level="variable",
            unit_level="unit",
            level_separator="|",
            iterable_input=(
                gdf for _, gdf in in_emissions.groupby(["model", "scenario"])
            ),
            parallel_op_config=ParallelOpConfig.from_user_facing(
                progress=self.progress,
                max_workers=self.n_processes,
            ),
        )
        res_d = defaultdict(list)
        for res_ms in res_g:
            if self.run_checks:
                assert_no_nans_in_res_ms(res_ms)
                assert_column_type_unchanged_in_res_ms(
                    res_ms, in_column_type=in_emissions.columns.dtype
                )

            for k, v in asdict(res_ms).items():
                res_d[k].append(v)

            complete_index_ms = get_required_ceds_index(
                world_region=self.world_region,
                model_regions=res_ms.gridding_workflow_emissions.index.get_level_values(
                    "region"
                ).difference([self.world_region]),
            )
            assert_all_groups_are_complete(
                res_ms.gridding_workflow_emissions, complete_index=complete_index_ms
            )

        res = CMIP7ScenarioMIPPreProcessingResult(
            **{k: pd.concat(v) for k, v in res_d.items()}
        )
        if self.run_checks:
            # Check we didn't lose anything on the way
            gridded_emisssions_sectoral_regional_sum = (
                get_gridded_emissions_sectoral_regional_sum(
                    res.gridding_workflow_emissions,
                    time_name="year",
                    region_level="region",
                    world_region=self.world_region,
                )
            )
            in_emissions_totals_to_compare_to = multi_index_lookup(
                in_emissions,
                gridded_emisssions_sectoral_regional_sum.index,
            )
            assert_frame_equal(
                in_emissions_totals_to_compare_to,
                gridded_emisssions_sectoral_regional_sum,
                rtol=self.rtol_internal_consistency,
                atol=self.atol_internal_consistency,
            )

            # Check internal consistency too
            reaggreated_gridded_emissions = aggregate_gridding_workflow_emissions_to_global_workflow_emissions(  # noqa: E501
                gridding_emissions=res.gridding_workflow_emissions,
                world_region=self.world_region,
                time_name="year",
                region_level="region",
                global_workflow_co2_fossil_sector_iamc="Energy and Industrial Processes",  # noqa: E501
                global_workflow_co2_biosphere_sector_iamc="AFOLU",
            )
            assert_frame_equal(
                multi_index_lookup(
                    res.global_workflow_emissions_raw_names,
                    reaggreated_gridded_emissions.index,
                ),
                reaggreated_gridded_emissions,
                rtol=self.rtol_internal_consistency,
                atol=self.atol_internal_consistency,
            )

        return res
