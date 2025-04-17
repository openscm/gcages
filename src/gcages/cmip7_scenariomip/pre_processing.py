"""
Pre-processing part of the workflow
"""

from __future__ import annotations

import multiprocessing
import re
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Callable, Protocol

import pandas as pd
from attrs import define
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_index_levels,
    assert_index_is_multiindex,
)
from gcages.exceptions import MissingOptionalDependencyError
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

if TYPE_CHECKING:
    import pyam


def split_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the variable into table, gas and sector

    This function is meant for internal use only.
    It is not subject to the usual set of tests or checks.

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] to work on

    Returns
    -------
    :
        `df` with the variable index level split into table, gas and sector
    """
    try:
        from pandas_indexing.core import extractlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "split_variable", requirement="pandas_indexing"
        ) from exc

    res = extractlevel(df, variable="{table}|{gas}|{sector}")

    return res


def combine_to_make_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine table, gas and sector to make a variable index level

    This function is meant for internal use only.
    It is not subject to the usual set of tests or checks.

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] to work on

    Returns
    -------
    :
        `df` with the variable index level created from table, gas and sector
    """
    try:
        from pandas_indexing.core import formatlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "combine_to_make_variable", requirement="pandas_indexing"
        ) from exc

    res = formatlevel(df, variable="{table}|{gas}|{sector}", drop=True)

    return res


def is_ceds_region_sector_level_relevant_variable(
    v: str,
    relevant_species: tuple[str, ...] = (
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
    ),
) -> bool:
    """
    Check if a variable is relevant for CEDS region-sector harmonisation

    Parameters
    ----------
    v
        Variable to check

    relevant_species
        Relevant species for CEDS region-sector harmonisation

        If any of these appear in `v`, we assume that the variable is relevant

    Returns
    -------
    :
        `True` if the variable is relevant for CEDS region-sector harmonisation,
        otherwise `False`
    """
    return any(v_name in v for v_name in relevant_species)


def process_transport_variables(  # noqa: PLR0913
    df: pd.DataFrame,
    aviation_sector_out: str = "Aircraft",
    transportation_sector_out: str = "Transportation Sector",
    aviation_domestic_sector_in: str = "Energy|Demand|Transportation|Domestic Aviation",
    aviation_international_sector_in: str = "Energy|Demand|Bunkers|International Aviation",  # noqa: E501
    transportation_sector_in: str = "Energy|Demand|Transportation",
) -> pd.DataFrame:
    """
    Process the transport variables into the convention needed

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] to process

    aviation_sector_out
        Label to use for the output aviation sector

    transportation_sector_out
        Label to use for the output transport sector

    aviation_domestic_sector_in
        Label to assume for the input domestic aviation sector

    aviation_international_sector_in
        Label to assume for the input international aviation sector

    transportation_sector_in
        Label to assume for the input transport sector

    Returns
    -------
    :
        `df` with the following changes:

        1. Domestic and international aviation
           are aggregated to a single aviation sector
        1. Domestic aviation is subtracted from transportation
        1. The input transportation sector is dropped
    """
    df_split = split_variable(df)

    df_stacked = df_split.unstack("sector").stack("year", future_stack=True)

    df_stacked[aviation_sector_out] = (
        df_stacked[aviation_domestic_sector_in]
        + df_stacked[aviation_international_sector_in]
    )
    df_stacked[transportation_sector_out] = (
        df_stacked[transportation_sector_in] - df_stacked[aviation_domestic_sector_in]
    )

    # Should we also consider dropping the aviation split for consistency?
    keep_sectors = df_stacked.columns.difference([transportation_sector_in])
    res = combine_to_make_variable(
        df_stacked[keep_sectors].unstack("year").stack("sector", future_stack=True)
    )

    return res


def aggregate_industry_sector(
    df: pd.DataFrame,
    sector_out: str = "Industrial Sector",
    sectors_to_aggregate: tuple[str, ...] = (
        "Energy|Demand|Industry",
        "Energy|Demand|Other Sector",
        "Industrial Processes",
        "Other",
    ),
) -> pd.DataFrame:
    """
    Aggregate the industry sector from its component sectors

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] in which to aggregate the industry sector

    sector_out
        Name to use for the output sector

    sectors_to_aggregate
        Sectors to aggregate to create `sector_out`


    Returns
    -------
    :
        `df` with the industrial sector included
    """
    df_split = split_variable(df)

    df_stacked = df_split.unstack("sector").stack("year", future_stack=True)
    df_stacked[sector_out] = df_stacked[list(sectors_to_aggregate)].sum(axis="columns")

    res = combine_to_make_variable(
        df_stacked.unstack("year").stack("sector", future_stack=True)
    )

    return res


def aggregate_ceds_like_agriculture(
    df: pd.DataFrame,
    sector_out: str = "CEDS Agriculture",
    sectors_to_aggregate: tuple[str, ...] = (
        "AFOLU|Agriculture",
        "AFOLU|Land|Land Use and Land-Use Change",
        "AFOLU|Land|Harvested Wood Products",
        "AFOLU|Land|Other",
        "AFOLU|Land|Wetlands",
    ),
) -> pd.DataFrame:
    """
    Aggregate a CEDS-like agriculture sector from its component sectors

    The default values for `sectors_to_aggregate` are imperfect
    but the best we have for now.
    We don't have a better source to harmonise to than CEDS
    (except for CO2 but ESMs don't use CO2 LULUCF
    emissions as input anyway, they use land-use change patterns,
    so this doesn't matter).
    For SCMs supported by OpenSCM-Runner,
    this mismatch also doesn't matter as it all gets rolled up to
    "AFOLU" anyway because SCMs aren't able to handle the difference
    between e.g. wood harvest and removals due to LULUCF.

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] in which to aggregate the industry sector

    sector_out
        Name to use for the output sector

    sectors_to_aggregate
        Sectors to aggregate to create `sector_out`


    Returns
    -------
    :
        `df` with the industrial sector included
    """
    df_split = split_variable(df)

    df_stacked = df_split.unstack("sector").stack("year", future_stack=True)
    df_stacked[sector_out] = df_stacked[list(sectors_to_aggregate)].sum(axis="columns")

    res = combine_to_make_variable(
        df_stacked.unstack("year").stack("sector", future_stack=True)
    )

    return res


DEFAULT_CEDS_RENAMINGS = {
    "Energy|Supply": "Energy Sector",
    "Industrial Sector": "Industrial Sector",  # assumes pre-processig
    "Energy|Demand|Residential and Commercial and AFOFI": "Residential Commercial Other",  # noqa: E501
    "Product Use": "Solvents Production and Application",
    "Transportation Sector": "Transportation Sector",
    "Waste": "Waste",
    "Aircraft": "Aircraft",
    "Energy|Demand|Bunkers|International Shipping": "International Shipping",
    "CEDS Agriculture": "Agriculture",
    "AFOLU|Agricultural Waste Burning": "Agricultural Waste Burning",
    "AFOLU|Land|Fires|Forest Burning": "Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning": "Grassland Burning",
    "AFOLU|Land|Fires|Peat Burning": "Peat Burning",
}
"""
Default renamigns used by [rename_and_cut_to_ceds_aligned_sectors][]
"""


def rename_and_cut_to_ceds_aligned_sectors(
    df: pd.DataFrame,
    renamings: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Rename sectors to CEDS sectors and only keep the renamed sectors

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] to work on

    renamings
        Renamings to apply

        If not supplied, we use [DEFAULT_CEDS_RENAMINGS]

    Returns
    -------
    :
        `df` with only renamed sectors
    """
    if renamings is None:
        renamings = DEFAULT_CEDS_RENAMINGS

    df_stacked = split_variable(df).unstack("sector").stack("year", future_stack=True)

    # TODO: remove this
    if "AFOLU|Land|Fires|Peat Burning" not in df_stacked:
        df_stacked["AFOLU|Land|Fires|Peat Burning"] = 0.0

    renamed = df_stacked.rename(renamings, axis="columns", errors="raise")

    res = combine_to_make_variable(
        renamed[renamings.values()].unstack("year").stack("sector", future_stack=True)
    )

    return res


def create_global_workflow_input_from_region_sector_input(  # noqa: PLR0913
    region_sector_df: pd.DataFrame,
    level_separator: str = "|",
    n_separators_for_single_sector: int = 2,
    world_region: str = "World",
    co2_fossil_sectors: tuple[str, ...] = (
        "Energy Sector",
        "Industrial Sector",
        "Residential Commercial Other",
        "Solvents Production and Application",
        "Transportation Sector",
        "Waste",
        "Aircraft",
        "International Shipping",
    ),
    co2_fossil_out_name: str = "Emissions|CO2|Energy and Industrial Processes",
    co2_afolu_out_name: str = "Emissions|CO2|AFOLU",
) -> pd.DataFrame:
    """
    Create input for the global workflow

    Parameters
    ----------
    region_sector_df
        [pd.DataFrame][pandas.DataFrame] with region-sector harmonisation information

    level_separator
        Separator used between levels in variable strings

    n_separators_for_single_sector
        Number of separators in a variable name when there is a single sector

    world_region
        Name of the world/world total region

    co2_fossil_sectors
        Sectors whose emissions originate from fossil CO2

    co2_fossil_out_name
        Name of the CO2 fossil-reservoir emissions output variable

    co2_afolu_out_name
        Name of the CO2 AFOLU-reservoir (i.e. biosphere) emissions output variable

    Returns
    -------
    :
        Output that can be used with the global-based workflow
    """
    region_sector_df_variables = region_sector_df.index.get_level_values(
        "variable"
    ).unique()
    not_single_sector = region_sector_df_variables[
        region_sector_df_variables.str.count(re.escape(level_separator))
        != n_separators_for_single_sector
    ]
    if not not_single_sector.empty:
        msg = (
            "The following variables in `region_sector_df` don't have a single sector"
            f"{not_single_sector}"
        )
        raise AssertionError(msg)

    if "World" in region_sector_df.index.get_level_values("region"):
        raise AssertionError

    def get_sum_over_sectors_and_regions(idf: pd.DataFrame) -> pd.DataFrame:
        return (
            update_index_levels_func(
                idf,
                dict(
                    variable=lambda x: level_separator.join(
                        x.split(level_separator)[:n_separators_for_single_sector]
                    ),
                    region=lambda x: world_region,
                ),
            )
            .groupby(idf.index.names)
            .sum()
        )

    region_sector_world_totals = get_sum_over_sectors_and_regions(region_sector_df)
    non_co2 = region_sector_world_totals.loc[
        region_sector_world_totals.index.get_level_values("variable") != "Emissions|CO2"
    ]

    region_sector_df_variable_level = region_sector_df.index.get_level_values(
        "variable"
    )
    co2_idx = region_sector_df_variable_level.str.startswith("Emissions|CO2")
    fossil_sector_idx = region_sector_df_variable_level.map(
        lambda x: any(x.endswith(s) for s in co2_fossil_sectors)
    )

    co2_fossil_components = region_sector_df.loc[co2_idx & fossil_sector_idx]
    co2_fossil = update_index_levels_func(
        get_sum_over_sectors_and_regions(co2_fossil_components),
        dict(variable=lambda x: co2_fossil_out_name),
    )

    # Doing it like this means we get all the burning in here too.
    # That is probably double counting,
    # although simple climate models don't have interactive fire emissions
    # so maybe it is fine.
    co2_afolu_components = region_sector_df.loc[co2_idx & ~fossil_sector_idx]
    co2_afolu = update_index_levels_func(
        get_sum_over_sectors_and_regions(co2_afolu_components),
        dict(variable=lambda x: co2_afolu_out_name),
    )

    res = pd.concat(
        [
            v.reorder_levels(co2_afolu.index.names)
            for v in [co2_fossil, co2_afolu, non_co2]
        ]
    )

    return res


def create_global_workflow_input_from_raw_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create global workflow input from raw input

    This should only be passed input that is not created
    from the region-sector relevant input

    Parameters
    ----------
    df
        Raw input

    Returns
    -------
    :
        Global workflow input derived from `df`
    """
    # For now, we can't use these.
    # If we figure out downscaling, we could
    # (but likely to be model-dependent hence hard to include here).
    out = strip_pint_incompatible_characters_from_units(
        df.loc[~df.index.get_level_values("unit").str.contains("equiv")],
        units_index_level="unit",
    )

    return out


class InternalConsistencyError(ValueError):
    """
    Raised when there is an internal consistency issue
    """

    def __init__(
        self,
        indf: pd.DataFrame,
        reporting_issues: pd.DataFrame,
        dsd: DataStructureDefinitionLike,
    ) -> None:
        # Continue from here:
        # - create a helpful summary which identifies:
        #   - failure points (just stack year from reporting_issues)
        #   - the variables that dsd includes in components
        #   - the variables that indf actually includes
        #   - the difference between the two above
        raise NotImplementedError
        error_msg = (
            "The DataFrame is not complete. "
            f"The following expected levels are missing:\n{missing}\n"
            f"The complete index expected for each level is:\n"
            f"{complete_index.to_frame(index=False)}"
        )
        super().__init__(error_msg)


class DataStructureDefinitionLike(Protocol):
    """
    Object that behaves like the nomenclature package's DataStructureDefinition class
    """

    def check_aggregate(self, df: pyam.IamDataFrame) -> pd.DataFrame | None:
        """
        Check aggregate along the variable hierarchy

        Parameters
        ----------
        df
            Data to check

        Returns
        -------
        :
            List of errors if there are any, otherwise `None`
        """


def assert_data_is_internally_consistent(
    indf: pd.DataFrame,
    dsd: DataStructureDefinitionLike,
    rtol: float = 1e-3,
    atol: float = 1e-8,
) -> None:
    try:
        import pyam
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "assert_data_is_internally_consistent", requirement="pyam"
        ) from exc

    reporting_issues = dsd.check_aggregate(
        pyam.IamDataFrame(indf), rtol=rtol, atol=atol
    )
    if reporting_issues is None:
        return

    raise InternalConsistencyError(
        indf=indf,
        reporting_issues=reporting_issues,
        dsd=dsd,
    )


@define
class CMIP7ScenarioMIPPreProcessingResult:
    """
    Result of pre-processing with [CMIP7ScenarioMIPPreProcessor][]

    This is more complex than normal,
    because we need to support both the 'normal' global path
    and harmonising at the region-sector level.
    """

    global_workflow_emissions: pd.DataFrame
    """
    Emissions that can be used with the 'normal' global workflow
    """

    global_workflow_emissions_gcages: pd.DataFrame
    """
    Emissions that can be used with the 'normal' global workflow using gcages naming
    """

    region_sector_workflow_emissions: pd.DataFrame
    """
    Emissions that can be used with the 'normal' global workflow
    """

    reaggregated_emissions: pd.DataFrame
    """
    Emissions that have been reaggregated, but otherwise not altered from the input
    """


@define
class CMIP7ScenarioMIPPreProcessor:
    """
    Pre-processor for CMIP7's ScenarioMIP
    """

    is_region_sector_relevant_variable: Callable[[str], bool] = (
        is_ceds_region_sector_level_relevant_variable
    )
    """
    Function to use to determine
    whether a variable is relevant for region-sector harmonisation or not
    """
    reprocess_transport_variables: Callable[[pd.DataFrame], pd.DataFrame] = (
        process_transport_variables
    )
    """
    Function to use to re-process the transport variables
    """

    calculate_industrial_sector: Callable[[pd.DataFrame], pd.DataFrame] = (
        aggregate_industry_sector
    )
    """
    Function to use to calculate the industrial sector
    """

    calculate_ceds_like_agriculture_sector: Callable[[pd.DataFrame], pd.DataFrame] = (
        aggregate_ceds_like_agriculture
    )
    """
    Function to use to calculate a CEDS-like agriculture sector
    """

    convert_to_and_extract_ceds_sectors: Callable[[pd.DataFrame], pd.DataFrame] = (
        rename_and_cut_to_ceds_aligned_sectors
    )
    """
    Function to use to convert to CEDS sectors and extract only CEDS sectors

    Called after reaggregation has already been done
    """

    convert_region_sector_to_global_workflow_input: Callable[
        [pd.DataFrame], pd.DataFrame
    ] = create_global_workflow_input_from_region_sector_input
    """
    Function to use to get the global workflow input that comes from region-sector input

    Called after processing to region-sector input has been done
    """

    convert_raw_to_global_workflow_input: Callable[[pd.DataFrame], pd.DataFrame] = (
        create_global_workflow_input_from_raw_input
    )
    """
    Function to use to get the global workflow input that comes from raw input

    This is applied on the input emissions except for the bits
    which are deemed relevant for region-sector level stuff.
    """

    world_region: str = "World"
    """
    String that identifies the world (i.e. global total) region
    """

    data_structure_definition: DataStructureDefinitionLike | None = None
    """
    Data structure definition
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
            assert_has_index_levels(in_emissions, ["variable", "unit"])

            if self.data_structure_definition is None:
                msg = (
                    "`self.data_structure_definition` must not be None "
                    "to run the checks"
                )
                raise TypeError(msg)

            assert_data_is_internally_consistent(
                in_emissions, self.data_structure_definition
            )

        if in_emissions.columns.name != "year":
            # Make later processing easier without annoying users
            in_emissions = in_emissions.copy()
            in_emissions.columns.name = "year"

        world_data_idx = (
            in_emissions.index.get_level_values("region") == self.world_region
        )
        region_sector_relevant_variable_idx = in_emissions.index.get_level_values(
            "variable"
        ).map(self.is_region_sector_relevant_variable)

        in_emissions_region_sector_relevant = in_emissions.loc[
            region_sector_relevant_variable_idx & ~world_data_idx
        ]
        in_emissions_not_region_sector_relevant = in_emissions.loc[
            ~region_sector_relevant_variable_idx & world_data_idx
        ]

        reaggregated_emissions = self.calculate_ceds_like_agriculture_sector(
            self.calculate_industrial_sector(
                self.reprocess_transport_variables(in_emissions_region_sector_relevant)
            )
        )

        region_sector_workflow_emissions = self.convert_to_and_extract_ceds_sectors(
            reaggregated_emissions
        )

        global_workflow_emissions_from_region_sector = (
            self.convert_region_sector_to_global_workflow_input(
                region_sector_df=region_sector_workflow_emissions
            )
        )
        global_workflow_emissions_from_raw = self.convert_raw_to_global_workflow_input(
            df=in_emissions_not_region_sector_relevant
        )

        global_workflow_emissions = pd.concat(
            [
                global_workflow_emissions_from_region_sector,
                global_workflow_emissions_from_raw.reorder_levels(
                    global_workflow_emissions_from_region_sector.index.names
                ),
            ]
        )

        global_workflow_emissions_gcages = update_index_levels_func(
            global_workflow_emissions,
            {
                "variable": partial(
                    convert_variable_name,
                    from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            },
        )

        return CMIP7ScenarioMIPPreProcessingResult(
            global_workflow_emissions=global_workflow_emissions,
            global_workflow_emissions_gcages=global_workflow_emissions_gcages,
            region_sector_workflow_emissions=region_sector_workflow_emissions,
            reaggregated_emissions=reaggregated_emissions,
        )
