"""
Definition of the pre-processor class
"""

from __future__ import annotations

import multiprocessing
from collections import defaultdict
from functools import partial

import pandas as pd
from attrs import asdict, define, field
from pandas_openscm.grouping import groupby_except
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.indexing import multi_index_lookup, multi_index_match
from pandas_openscm.parallelisation import ParallelOpConfig, apply_op_parallel_progress

from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_index_levels,
    assert_index_is_multiindex,
    assert_only_working_on_variable_unit_region_variations,
)
from gcages.cmip7_scenariomip.pre_processing.assertions import (
    assert_column_type_unchanged_in_res_ms,
    assert_data_has_required_internal_consistency,
    assert_data_is_compatible_with_pre_processing,
    assert_no_nans_in_res_ms,
)
from gcages.cmip7_scenariomip.pre_processing.completeness import (
    get_all_model_region_index_input,
    get_all_world_index_input,
    get_required_model_region_index_gridding,
    get_required_world_index_gridding,
)
from gcages.cmip7_scenariomip.pre_processing.constants import (
    AGRICULTURE_SECTOR_REAGGREGATED,
    CO2_BIOSPHERE_SECTORS_GRIDDING,
    CO2_FOSSIL_SECTORS_GRIDDING,
    INDUSTRIAL_SECTOR_REAGGREGATED,
    OPTIONAL_AGRICULTURE_SECTOR_GRIDDING_COMPONENTS_INPUT,
    OPTIONAL_INDUSTRIAL_SECTOR_GRIDDING_COMPONENTS_INPUT,
    REAGGREGATED_TO_GRIDDING_SECTOR_MAP_MODEL_REGION,
    REAGGREGATED_TO_GRIDDING_SECTOR_MAP_WORLD,
    REQUIRED_AGRICULTURE_SECTOR_GRIDDING_COMPONENTS_INPUT,
    REQUIRED_INDUSTRIAL_SECTOR_GRIDDING_COMPONENTS_INPUT,
)
from gcages.cmip7_scenariomip.pre_processing.sector_cols import (
    aggregate_sector,
    convert_to_global_workflow_emissions,
    reclassify_aviation_emissions,
)
from gcages.completeness import assert_all_groups_are_complete
from gcages.index_manipulation import (
    combine_sectors,
    combine_species,
    set_new_single_value_levels,
    split_sectors,
)
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import assert_frame_equal
from gcages.units_helpers import strip_pint_incompatible_characters_from_units


@define
class CMIP7ScenarioMIPPreProcessingResult:
    """
    Result of pre-processing with [CMIP7ScenarioMIPPreProcessor][(m).]

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


def do_pre_processing(  # noqa: PLR0913
    indf: pd.DataFrame,
    world_region: str,
    model_regions: tuple[str, ...],
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

    used_in_gridding_index = get_all_world_index_input(
        world_region=world_region,
        region_level=region_level,
        variable_level=variable_level,
    ).append(
        get_all_model_region_index_input(
            model_regions,
            region_level=region_level,
            variable_level=variable_level,
        )
    )
    used_in_gridding_locator = multi_index_match(
        indf_clean_units.index, used_in_gridding_index
    )
    indf_used_in_gridding = indf_clean_units.loc[used_in_gridding_locator]

    world_locator = (
        indf_used_in_gridding.index.get_level_values(region_level) == world_region
    )
    # DataFrames are named by whether they have
    # region and sector dimensions or just sector
    # and whether their columns are sector or time.
    region_sector_df_sector_col = (
        split_sectors(
            indf_used_in_gridding.loc[~world_locator],
            middle_level="species",
            bottom_level="sectors",
            level_separator=level_separator,
        )
        .stack()
        .unstack("sectors")
    )
    sector_df_sector_col = (
        split_sectors(
            indf_used_in_gridding.loc[world_locator].reset_index("region", drop=True),
            middle_level="species",
            bottom_level="sectors",
            level_separator=level_separator,
        )
        .stack()
        .unstack("sectors")
    )

    region_sector_df_sector_col, sector_df_sector_col = reclassify_aviation_emissions(
        region_sector_df=region_sector_df_sector_col,
        sector_df=sector_df_sector_col,
        region_level=region_level,
        copy=False,
    )

    region_sector_df_sector_col = aggregate_sector(
        region_sector_df_sector_col,
        sector_out=INDUSTRIAL_SECTOR_REAGGREGATED,
        sector_components=[
            *REQUIRED_INDUSTRIAL_SECTOR_GRIDDING_COMPONENTS_INPUT,
            *OPTIONAL_INDUSTRIAL_SECTOR_GRIDDING_COMPONENTS_INPUT,
        ],
        time_name=time_name,
        allow_missing=list(OPTIONAL_INDUSTRIAL_SECTOR_GRIDDING_COMPONENTS_INPUT),
    )

    region_sector_df_sector_col = aggregate_sector(
        region_sector_df_sector_col,
        sector_out=AGRICULTURE_SECTOR_REAGGREGATED,
        sector_components=[
            *REQUIRED_AGRICULTURE_SECTOR_GRIDDING_COMPONENTS_INPUT,
            *OPTIONAL_AGRICULTURE_SECTOR_GRIDDING_COMPONENTS_INPUT,
        ],
        allow_missing=list(OPTIONAL_AGRICULTURE_SECTOR_GRIDDING_COMPONENTS_INPUT),
        time_name=time_name,
    )
    region_sector_gridding_df_sector_col = region_sector_df_sector_col.rename(
        REAGGREGATED_TO_GRIDDING_SECTOR_MAP_MODEL_REGION, axis="columns", errors="raise"
    )
    sector_gridding_df_sector_col = sector_df_sector_col.rename(
        REAGGREGATED_TO_GRIDDING_SECTOR_MAP_WORLD, axis="columns", errors="raise"
    )

    cs = partial(
        combine_sectors,
        middle_level="species",
        bottom_level="sectors",
        level_separator=level_separator,
    )
    gridding_workflow_emissions_region_sector = cs(
        region_sector_gridding_df_sector_col.stack().unstack(time_name),
    )
    gridding_workflow_emissions_sector = set_new_single_value_levels(
        cs(
            sector_gridding_df_sector_col.stack().unstack(time_name),
            level_separator=level_separator,
        ),
        {region_level: world_region},
    )

    gridding_workflow_emissions = pd.concat(
        [
            gridding_workflow_emissions_region_sector,
            gridding_workflow_emissions_sector.reorder_levels(
                gridding_workflow_emissions_region_sector.index.names
            ),
        ]
    )

    global_workflow_emissions_from_gridding_emissions = (
        convert_to_global_workflow_emissions(
            region_sector_df=region_sector_gridding_df_sector_col,
            sector_df=sector_gridding_df_sector_col,
            time_name=time_name,
            region_level=region_level,
            world_region=world_region,
            global_workflow_co2_fossil_sector="Energy and Industrial Processes",
            global_workflow_co2_biosphere_sector="AFOLU",
            co2_fossil_sectors=CO2_FOSSIL_SECTORS_GRIDDING,
            co2_biosphere_sectors=CO2_BIOSPHERE_SECTORS_GRIDDING,
            species_level="species",
            co2_name="CO2",
        )
    )

    species_in_gridding = region_sector_gridding_df_sector_col.index.get_level_values(
        "species"
    ).unique()
    variable_starts_in_gridding = tuple(f"Emissions|{s}" for s in species_in_gridding)
    global_workflow_emissions_not_from_gridding_emissions = indf_clean_units.loc[
        ~indf_clean_units.index.get_level_values("variable").str.startswith(
            variable_starts_in_gridding
        )
    ]
    # Can't use these yet
    # TODO: implement support for baskets
    global_workflow_emissions_not_from_gridding_emissions = global_workflow_emissions_not_from_gridding_emissions.loc[  # noqa: E501
        ~global_workflow_emissions_not_from_gridding_emissions.index.get_level_values(
            "unit"
        ).str.contains("equiv")
    ]

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


@define
class CMIP7ScenarioMIPPreProcessor:
    """
    Pre-processor for CMIP7's ScenarioMIP

    For more details of the logic, see [gcages.cmip7_scenariomip.pre_processing][].
    """

    tols_internal_consistency: dict[str, dict[str, float]] = field()
    """
    Tolerances to apply when checking the internal consistency of the data

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
    Should progress bars be shown?
    """

    n_processes: int | None = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to `None` to process in serial.
    """

    @tols_internal_consistency.default
    def default_tols_internal_consistency(self) -> dict[str, dict[str, float]]:
        """
        Get default tolerances for internal consistency checks
        """
        return {
            "Emissions|BC": dict(rtol=1e-3, atol=1e-6),
            "Emissions|CH4": dict(rtol=1e-3, atol=1e-6),
            "Emissions|CO": dict(rtol=1e-3, atol=1e-6),
            # Higher absolute tolerance because of reporting units
            "Emissions|CO2": dict(rtol=1e-3, atol=1.0),
            "Emissions|NH3": dict(rtol=1e-3, atol=1e-6),
            "Emissions|NOx": dict(rtol=1e-3, atol=1e-6),
            "Emissions|OC": dict(rtol=1e-3, atol=1e-6),
            "Emissions|Sulfur": dict(rtol=1e-3, atol=1e-6),
            "Emissions|VOC": dict(rtol=1e-3, atol=1e-6),
            "Emissions|N2O": dict(rtol=1e-3, atol=1e-6),
        }

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

        assumed_model_regions = [
            r
            for r in in_emissions.index.get_level_values("region").unique()
            if r != self.world_region
        ]
        if self.run_checks:
            if in_emissions.columns.name != "year":
                msg = "The input emissions' column name should be 'year'"
                raise AssertionError(msg)

            for _, msdf in in_emissions.groupby(["model", "scenario"]):
                msdf_drop_all_nan_times = msdf.dropna(how="all", axis="columns")

                assert_data_is_compatible_with_pre_processing(
                    msdf_drop_all_nan_times,
                    world_region=self.world_region,
                    region_level="region",
                    variable_level="variable",
                    model_regions=assumed_model_regions,
                )

                assert_data_has_required_internal_consistency(
                    msdf_drop_all_nan_times,
                    model_regions=assumed_model_regions,
                    world_region=self.world_region,
                    region_level="region",
                    variable_level="variable",
                    tols=self.tols_internal_consistency,
                    # TODO: consider making these passable
                    level_separator="|",
                    n_levels_for_total=1,
                )

        res_g = apply_op_parallel_progress(
            func_to_call=do_pre_processing,
            world_region=self.world_region,
            model_regions=assumed_model_regions,
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

            complete_index_ms = get_required_world_index_gridding(
                world_region=self.world_region,
                region_level="region",
                variable_level="variable",
            ).append(
                get_required_model_region_index_gridding(
                    model_regions=assumed_model_regions,
                    region_level="region",
                    variable_level="variable",
                )
            )
            assert_all_groups_are_complete(
                res_ms.gridding_workflow_emissions, complete_index=complete_index_ms
            )

        res = CMIP7ScenarioMIPPreProcessingResult(
            **{k: pd.concat(v) for k, v in res_d.items()}
        )
        if self.run_checks:
            # Check internal consistency
            assert_data_has_required_internal_consistency(
                res.gridding_workflow_emissions,
                model_regions=assumed_model_regions,
                world_region=self.world_region,
                region_level="region",
                variable_level="variable",
                tols=self.tols_internal_consistency,
                # TODO: consider making these passable
                level_separator="|",
                n_levels_for_total=1,
            )

            # Check we didn't lose any mass on the way
            gridded_emisssions_sectoral_regional_sum = set_new_single_value_levels(
                combine_species(
                    groupby_except(
                        split_sectors(
                            res.gridding_workflow_emissions, bottom_level="sectors"
                        ),
                        ["region", "sectors"],
                    ).sum()
                ),
                {"region": self.world_region},
            )

            in_emissions_totals_to_compare_to = multi_index_lookup(
                in_emissions,
                gridded_emisssions_sectoral_regional_sum.index,
            )
            assert_frame_equal(
                in_emissions_totals_to_compare_to,
                gridded_emisssions_sectoral_regional_sum,
                # No tolerance as this should be exact
            )

        return res
