"""
Pre-processing part of the workflow
"""

from __future__ import annotations

import multiprocessing
from collections import defaultdict

import numpy as np
import pandas as pd
from attrs import define
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
from gcages.testing import assert_frame_equal

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

DOMESTIC_AVIATION_SECTOR_IAMC: str = "Energy|Demand|Transportation|Domestic Aviation"
"""
Assumed sector for the domestic aviation sector (IAMC naming convention)
"""

INTERNATIONAL_AVIATION_SECTOR_IAMC: str = "Energy|Demand|Bunkers|International Aviation"
"""
Assumed reporting for the international aviation sector (IAMC naming convention)
"""

TRANSPORTATION_SECTOR_IAMC: str = "Energy|Demand|Transportation"
"""
Assumed reporting for the transportation sector (IAMC naming convention)
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
    world_idxer = indf.index.get_level_values("region") == world_region
    world_data = indf.loc[world_idxer]
    regional_data = indf.loc[~world_idxer]

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


def assert_data_is_compatible_with_pre_processing(
    indf: pd.DataFrame, world_region: str, region_level: str
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

    assert_frame_equal(data_for_gridding_totals, indf_global_totals)


@define
class CMIP7ScenarioMIPPreProcessingResult:
    """
    Result of pre-processing with [CMIP7ScenarioMIPPreProcessor][]

    This has more components than normal,
    because we need to support both the 'normal' global path
    and harmonising at the region-sector level.
    """

    global_workflow_emissions: pd.DataFrame
    """
    Emissions that can be used with the 'normal' global workflow
    """

    global_workflow_emissions_iamc: pd.DataFrame
    """
    Emissions consistent with those that can be used with the 'normal' global workflow

    The difference is that these are reported with IAMC naming,
    which isn't compatible with our runners
    (so probably not what you want to use,
    but perhaps helpful for plotting and direct comparisons).
    """

    gridding_workflow_emissions: pd.DataFrame
    """
    Emissions that can be used with the gridding workflow
    """


def do_pre_processing(
    indf: pd.DataFrame, world_region: str
) -> CMIP7ScenarioMIPPreProcessingResult:
    split_data = split_world_and_regional_data(indf, world_region=world_region)
    breakpoint()


@define
class CMIP7ScenarioMIPPreProcessor:
    """
    Pre-processor for CMIP7's ScenarioMIP

    Like most pre-processing,
    this is quite locked up and context-specific.
    The logic also goes through multiple layers,
    which makes it hard to clearly identify
    how techniques like dependency injection could actually be used.
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

            assert_data_is_compatible_with_pre_processing(
                in_emissions, world_region=self.world_region, region_level="region"
            )

            if in_emissions.columns.name != "year":
                msg = "The input emissions' column name should be 'year'"
                raise AssertionError(msg)

        res_g = apply_op_parallel_progress(
            func_to_call=do_pre_processing,
            world_region=self.world_region,
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
                for attr in [
                    "global_workflow_emissions",
                    "gridding_workflow_emissions",
                ]:
                    if res_ms.isnull().any().any():
                        ms = (
                            res_ms.index.droplevels(
                                res_ms.index.names.difference(["model", "scenario"])
                            )
                            .drop_duplicates()
                            .tolist()
                        )
                        msg = f"NaNs in res.{attr} for {ms}"
                        raise AssertionError(msg)

            for attr in [
                "global_workflow_emissions",
                "global_workflow_emissions_iamc",
                "gridding_workflow_emissions",
            ]:
                res_d[attr].append(getattr(res_ms, attr))

        res = CMIP7ScenarioMIPPreProcessingResult(
            **{k: pd.concat(v) for k, v in res_d.items()}
        )
        if self.run_checks:
            assert_all_expected_variables_and_regions_included(
                res.gridding_workflow_emissions
            )
            assert_gridding_and_global_emissions_are_consistent(
                gridding_workflow_emissions=res.gridding_workflow_emissions,
                global_workflow_emissions_iamc=res.global_workflow_emissions_iamc,
            )

        return res
