"""
Definition of the pre-processor class
"""

from __future__ import annotations

import multiprocessing
from collections import defaultdict
from functools import partial
from typing import Protocol

import pandas as pd
from attrs import asdict, define
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.indexing import multi_index_lookup, multi_index_match
from pandas_openscm.parallelisation import ParallelOpConfig, apply_op_parallel_progress

from gcages.aggregation import get_region_sector_sum
from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_index_levels,
    assert_index_is_multiindex,
    assert_only_working_on_variable_unit_region_variations,
)
from gcages.cmip7_scenariomip.gridding_emissions import (
    CO2_BIOSPHERE_SECTORS_GRIDDING,
    CO2_FOSSIL_SECTORS_GRIDDING,
    get_complete_gridding_index,
    to_global_workflow_emissions,
)
from gcages.cmip7_scenariomip.pre_processing.reaggregation import (
    ReaggregatorBasic,
    ToCompleteResult,
)
from gcages.completeness import NotCompleteError, assert_all_groups_are_complete
from gcages.index_manipulation import split_sectors
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

    assumed_zero_emissions: pd.DataFrame | None
    """
    Emissions that were asssumed to be zero during the processing
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


def guess_reaggregator(
    indf: pd.DataFrame,
    region_level: str,
) -> ReaggregatorLike:
    """
    Guess the re-aggregator to use with a given dataset

    Parameters
    ----------
    indf
        Data for which to guess the re-aggregator

    region_level
        Region level in the data index

    Returns
    -------
    :
        Guessed re-aggregator

    Raises
    ------
    ValueError
        Re-aggregator could not be guessed for `indf`
    """
    assumed_model_regions = tuple(
        r for r in indf.index.get_level_values(region_level).unique() if r != "World"
    )
    errors_l = []
    for guess_cls in (ReaggregatorBasic,):
        guess = guess_cls(
            model_regions=assumed_model_regions, region_level=region_level
        )

        try:
            guess.assert_has_all_required_timeseries(indf)

        except NotCompleteError as exc:
            # Not a match
            errors_l.append(f"For {guess_cls}, error was:\n{exc}")
            continue

        else:
            return guess

    errors = "\n".join(errors_l)
    msg = (
        "Could not guess the reaggregator for the given input:\n"
        f"{indf}.\n"
        f"Errors:\n{errors}"
    )
    raise ValueError(msg)


def do_pre_processing(  # noqa: PLR0912, PLR0913, PLR0915
    indf: pd.DataFrame,
    reaggregator: ReaggregatorLike | None,
    time_name: str,
    run_checks: bool,
    world_gridding_sectors: tuple[str, ...] = ("Aircraft", "International Shipping"),
    table: tuple[str, ...] = ("Emissions", "Carbon Removal"),
    level_separator: str = "|",
    co2_fossil_sectors: tuple[str, ...] = CO2_FOSSIL_SECTORS_GRIDDING,
    co2_biosphere_sectors: tuple[str, ...] = CO2_BIOSPHERE_SECTORS_GRIDDING,
    co2_name: str = "CO2",
) -> CMIP7ScenarioMIPPreProcessingResult:
    """
    Do the pre-processing for a given scenario

    This only works on a single scenario at a time,
    to make verification and processing simpler.

    Parameters
    ----------
    indf
        Input data to process

    reaggregator
        Re-aggregator to use during the processing

    time_name
        Name of the time axis in `indf`

    run_checks
        Should checks be run during the processing?

        If you know what you're doing, you can turn these off for speed.

    world_gridding_sectors
        Sectors that should only be gridded at the world level

    table
        Name of the 'table' for emissions

        Used to process and create variable names

    level_separator
        Separator between levels in the variable names

    co2_fossil_sectors
        Sectors to assume have an origin in fossil CO2 reservoirs

        These should be provided in the gridding naming convention

    co2_biosphere_sectors
        Sectors to assume have an origin in biospheric CO2 reservoirs

        These should be provided in the gridding naming convention

    co2_name
        String that indicates emissions of CO2 in variable names

    Returns
    -------
    :
        Results of the pre-processing
    """
    assert_only_working_on_variable_unit_region_variations(indf)

    if reaggregator is None:
        # Levels we will guess
        region_level = "region"
        unit_level = "unit"
        variable_level = "variable"

    else:
        region_level = reaggregator.region_level
        unit_level = reaggregator.unit_level
        variable_level = reaggregator.variable_level

    if run_checks:
        assert_has_index_levels(
            indf,
            ["model", "scenario", region_level, unit_level, variable_level],
        )

    if reaggregator is None:
        reaggregator = guess_reaggregator(indf, region_level=region_level)

    indf_reported_times = indf.dropna(how="all", axis="columns")

    if run_checks:
        indf_reported_times_nan = indf_reported_times.isnull().any(axis="columns")
        if indf_reported_times_nan.any():
            issue_rows = indf.loc[indf_reported_times_nan, :]
            msg = f"NaNs after dropping unreported times:\n{issue_rows}"
            raise AssertionError(msg)

    indf_clean_units = strip_pint_incompatible_characters_from_units(
        indf_reported_times,
        units_index_level=reaggregator.unit_level,
    )

    if run_checks:
        reaggregator.assert_has_all_required_timeseries(indf_clean_units)
        reaggregator.assert_is_internally_consistent(indf_clean_units)

    to_complete_result = reaggregator.to_complete(indf_clean_units)
    gridding_workflow_emissions = reaggregator.to_gridding_sectors(
        to_complete_result.complete
    )

    if run_checks:
        if gridding_workflow_emissions.isnull().any().any():
            msg = "NaN in `gridding_workflow_emissions`"
            raise AssertionError(msg)

        if gridding_workflow_emissions.columns.dtype != indf.columns.dtype:
            msg = "Column type does not match input"
            raise AssertionError(msg)

        complete_index_gridding = get_complete_gridding_index(
            model_regions=reaggregator.model_regions,
            world_gridding_sectors=world_gridding_sectors,
            world_region=reaggregator.world_region,
            region_level=reaggregator.region_level,
            variable_level=reaggregator.variable_level,
            table=table,
            level_separator=level_separator,
        )
        assert_all_groups_are_complete(
            gridding_workflow_emissions, complete_index=complete_index_gridding
        )

        # Check we didn't lose any mass
        grss = partial(
            get_region_sector_sum,
            region_level=reaggregator.region_level,
            world_region=reaggregator.world_region,
        )
        gridded_emisssions_sectoral_regional_sum = grss(gridding_workflow_emissions)

        in_emissions_totals_to_compare_to = grss(
            # Make sure we only sum across the levels
            # that are useful for getting the total
            multi_index_lookup(
                indf, reaggregator.get_internal_consistency_checking_index()
            )
        )

        # The sign of the 'Carbon Removal' must be flipped to compare
        # (it gets changed in the reaggregation)
        mask = gridded_emisssions_sectoral_regional_sum.index.get_level_values(
            "variable"
        ).str.startswith("Carbon Removal")
        gridded_emisssions_sectoral_regional_sum.loc[mask] *= -1
        # To account for CDR in the sum
        cdr = in_emissions_totals_to_compare_to.xs(
            "Carbon Removal|CO2", level="variable"
        )
        emi = in_emissions_totals_to_compare_to.xs("Emissions|CO2", level="variable")
        # Sum
        co2_sum = emi + cdr
        # Assign the result back into the original DataFrame under "Emissions|CO2"
        for idx, row in co2_sum.iterrows():
            new_idx = (
                idx[:3] + ("Emissions|CO2",) + idx[3:]  # type: ignore
            )  # Rebuild full MultiIndex
            in_emissions_totals_to_compare_to.loc[new_idx] = row.values

        # No tolerance as this should be exact
        assert_frame_equal(
            gridded_emisssions_sectoral_regional_sum,
            in_emissions_totals_to_compare_to,
        )

    # Figure out the global workflow emissions
    global_workflow_emissions_from_gridding_emissions = to_global_workflow_emissions(
        gridding_workflow_emissions,
        time_name=time_name,
        region_level=reaggregator.region_level,
        world_region=reaggregator.world_region,
        # These have to be hard-coded to the IAM naming convention
        global_workflow_co2_fossil_sector="Energy and Industrial Processes",
        global_workflow_co2_biosphere_sector="AFOLU",
        co2_fossil_sectors=co2_fossil_sectors,
        co2_biosphere_sectors=co2_biosphere_sectors,
        co2_name=co2_name,
    )

    gwe_split = split_sectors(gridding_workflow_emissions, middle_level="species")
    species_from_gridding = tuple(gwe_split.index.get_level_values("species").unique())

    # Firstly drop out everything which was used for gridding
    indf_obviously_not_used_in_gridding = indf_clean_units.loc[
        ~multi_index_match(indf_clean_units.index, to_complete_result.complete.index)  # type: ignore
    ]

    # Now do the brute check on whatever is leftover
    def species_in_variable(variable: str, species: str, ls: str) -> bool:
        # ls: level separator
        # This mucking around is another illustration of the
        # issue with the data reporting format
        # (you need to check endswith and surrounded by the separator
        # to avoid accidental matches like VOC and OC)
        return variable.endswith(species) or (f"{ls}{species}{ls}" in variable)

    not_from_region_sector = [
        variable
        for variable in indf_obviously_not_used_in_gridding.index.get_level_values(
            variable_level
        ).unique()
        if not any(
            species_in_variable(variable, species=sg, ls=level_separator)
            for sg in species_from_gridding
        )
    ]
    global_workflow_emissions_not_from_gridding_emissions = indf_clean_units.loc[
        indf_clean_units.index.get_level_values(variable_level).isin(
            not_from_region_sector
        )
        # By definition, only want global emissions
        & (
            indf_clean_units.index.get_level_values(region_level)
            == reaggregator.world_region
        )
    ]
    # Can't use these yet
    # TODO: implement support for baskets
    global_workflow_emissions_not_from_gridding_emissions = global_workflow_emissions_not_from_gridding_emissions.loc[  # noqa: E501
        ~global_workflow_emissions_not_from_gridding_emissions.index.get_level_values(
            unit_level
        ).str.contains("equiv")
    ]

    global_workflow_emissions_raw_names = pd.concat(
        [
            df.reorder_levels(
                global_workflow_emissions_from_gridding_emissions.index.names
            )
            for df in [
                global_workflow_emissions_from_gridding_emissions,
                global_workflow_emissions_not_from_gridding_emissions,
            ]
        ]
    )

    if run_checks:
        if global_workflow_emissions_raw_names.isnull().any().any():
            msg = "NaN in `global_workflow_emissions_raw_names`"
            raise AssertionError(msg)

        if global_workflow_emissions_raw_names.columns.dtype != indf.columns.dtype:
            msg = "Column type does not match input"
            raise AssertionError(msg)

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
        assumed_zero_emissions=to_complete_result.assumed_zero,
        gridding_workflow_emissions=gridding_workflow_emissions,
        global_workflow_emissions=global_workflow_emissions,
        global_workflow_emissions_raw_names=global_workflow_emissions_raw_names,
    )

    return res


class ReaggregatorLike(Protocol):
    """
    Interface that can be used for re-aggregation
    """

    model_regions: tuple[str, ...]
    """Model regions to use while reaggregating"""

    region_level: str
    """Region level in the data index"""

    unit_level: str
    """Unit level in the data index"""

    variable_level: str
    """Variable level in the data index"""

    world_region: str
    """
    The value used when the data represents the sum over all regions

    (Having a value for this is odd,
    there should really just be no region level when data is the sum,
    but this is the data format used so we have to follow this convention.)
    """

    def assert_has_all_required_timeseries(self, indf: pd.DataFrame) -> None:
        """
        Assert that the data has all the required timeseries

        Parameters
        ----------
        indf
            Data to check

        Raises
        ------
        NotCompleteError
            `indf` is not complete
        """

    def assert_is_internally_consistent(self, indf: pd.DataFrame) -> None:
        """
        Assert that the data is internally consistent

        Parameters
        ----------
        indf
            Data to check

        Raises
        ------
        InternalConsistencyError
            The data is not internally consistent
        """

    def get_internal_consistency_checking_index(self) -> pd.MultiIndex:
        """
        Get the index which selects only data relevant for checking internal consistency

        Returns
        -------
        :
            Internal consistency checking index
        """

    def to_complete(self, raw: pd.DataFrame) -> ToCompleteResult:
        """
        Convert the raw data to complete data

        Parameters
        ----------
        raw
            Raw data

        Returns
        -------
        :
            To complete result
        """

    def to_gridding_sectors(self, indf: pd.DataFrame) -> pd.DataFrame:
        """
        Re-aggregate data to the sectors used for gridding

        Parameters
        ----------
        indf
            Data to re-aggregate

        Returns
        -------
        :
            Data re-aggregated to the gridding sectors
        """


@define
class CMIP7ScenarioMIPPreProcessor:
    """
    Pre-processor for CMIP7's ScenarioMIP

    For more details of the logic, see [gcages.cmip7_scenariomip.pre_processing][].
    """

    reaggregator: ReaggregatorLike | None = None
    """
    Re-aggregator to use when converting raw data to gridding sectors

    If not supplied, we guess the re-aggregator during processing
    """

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    world_gridding_sectors: tuple[str, ...] = ("Aircraft", "International Shipping")
    """
    Sectors that are only used for gridding at the world (i.e. regional sum) level
    """

    co2_fossil_sectors: tuple[str, ...] = CO2_FOSSIL_SECTORS_GRIDDING
    """
    Gridding sectors that are assumed to come from the fossil CO2 reservoir
    """

    co2_biosphere_sectors: tuple[str, ...] = CO2_BIOSPHERE_SECTORS_GRIDDING
    """
    Gridding sectors that are assumed to come from the biosphere CO2 reservoir
    """

    co2_name: str = "CO2"
    """
    Name used for CO2 in variable names
    """

    table: tuple[str, ...] = ("Emissions", "Carbon Removal")
    """
    The value used for the top level of variable names
    """

    level_separator: str = "|"
    """
    The separator between levels in variable names
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

            if in_emissions.columns.name != "year":
                msg = "The input emissions' column name should be 'year'"
                raise AssertionError(msg)

        res_g = apply_op_parallel_progress(
            func_to_call=do_pre_processing,
            reaggregator=self.reaggregator,
            time_name="year",
            run_checks=self.run_checks,
            world_gridding_sectors=self.world_gridding_sectors,
            table=self.table,
            level_separator=self.level_separator,
            co2_fossil_sectors=self.co2_fossil_sectors,
            co2_biosphere_sectors=self.co2_biosphere_sectors,
            co2_name=self.co2_name,
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
            for k, v in asdict(res_ms).items():
                if v is not None:
                    res_d[k].append(v)

        result_initialiser = {k: pd.concat(v) for k, v in res_d.items()}
        if "assumed_zero_emissions" not in result_initialiser:
            result_initialiser["assumed_zero_emissions"] = None

        res = CMIP7ScenarioMIPPreProcessingResult(**result_initialiser)

        return res
