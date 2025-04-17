"""
Pre-processing part of the workflow
"""

from __future__ import annotations

import multiprocessing
from typing import Callable

import pandas as pd
from attrs import define

from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_index_levels,
    assert_index_is_multiindex,
)
from gcages.exceptions import MissingOptionalDependencyError


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
        "Energy|Supply",
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

    region_sector_workflow_emissions: pd.DataFrame
    """
    Emissions that can be used with the 'normal' global workflow
    """


@define
class CMIP7ScenarioMIPPreProcessor:
    """
    Pre-processor for CMIP7's ScenarioMIP
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

        region_sector_workflow_emissions = self.reprocess_transport_variables(
            in_emissions
        )
        region_sector_workflow_emissions = self.calculate_industrial_sector(
            region_sector_workflow_emissions
        )
        global_workflow_emissions = None

        # Two returns:
        # - one for normal harmonisation i.e. everything already aggregated
        # - one for region-sector harmonisation gcages (TODO, requires renaming)
        # - one for region-sector harmonisation IAMC names
        return CMIP7ScenarioMIPPreProcessingResult(
            global_workflow_emissions=global_workflow_emissions,
            region_sector_workflow_emissions=region_sector_workflow_emissions,
        )
