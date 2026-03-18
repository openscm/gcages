"""
Infilling configuration and related things for the CMIP7 ScenarioMIP workflow
"""

from __future__ import annotations

import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip.harmonisation import (
    load_cmip7_scenariomip_historical_emissions,
)
from gcages.cmip7_scenariomip.scm_running import complete_index_reporting_names
from gcages.completeness import assert_all_groups_are_complete
from gcages.exceptions import MissingOptionalDependencyError
from gcages.harmonisation.common import assert_harmonised
from gcages.hashing import get_file_hash
from gcages.renaming import SupportedNamingConventions, convert_variable_name

if TYPE_CHECKING:
    import silicone


@dataclass
class CMIP7ScenarioMIPInfilledScenarios:
    """
    Result of infilling

    """

    silicone: pd.DataFrame | None = None
    """
    Emissions silicone infilled
    """
    wmo: pd.DataFrame | None = None
    """
    Emissions wmo infilled
    """
    scaled: pd.DataFrame | None = None
    """
    Emissions scaled infilled
    """
    complete: pd.DataFrame | None = None
    """
    Complete infilled emissions
    """

    def _add(self, name: str, df: pd.DataFrame) -> None:
        """
        Add an infilled emissions DataFrame as an attribute.

        Parameters
        ----------
        name
            Attribute name to assign.
        df
            Emissions dataset to attach.
        """
        setattr(self, name, df)


@dataclass
class InfillingSources:
    """
    Source needed for infilling
    """

    infilling_db: pd.DataFrame
    """
    Infilling database
    """
    historical_emissions: pd.DataFrame
    """
    Historical emissions
    """
    cmip7_ghg_inversions: pd.DataFrame
    """
    Green house gasses inversions
    """


def get_silicone_based_infiller(  # type: ignore # silicone has no type hints
    infilling_db: pd.DataFrame,
    follower_variable: str,
    lead_variables: list[str],
    silicone_db_cruncher: silicone.database_crunchers.base._DatabaseCruncher,
    derive_relationship_kwargs: dict[str, Any] | None = None,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Get an infiller based on silicone

    Parameters
    ----------
    infilling_db
        Infilling database

    follower_variable
        The variable to infill

    lead_variables
        The variables used to infill `follower_variable`

    silicone_db_cruncher
        Silicone cruncher to use

    derive_relationship_kwargs
        Passed to `silicone_db_cruncher.derive_relationship`

    Returns
    -------
    :
        Function which can be used to infill `follower_variable` in scenarios
    """
    try:
        import pyam
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_silicone_based_infiller", requirement="pyam"
        ) from exc

    if derive_relationship_kwargs is None:
        derive_relationship_kwargs = {}

    silicone_infiller = silicone_db_cruncher(
        pyam.IamDataFrame(infilling_db)
    ).derive_relationship(
        variable_follower=follower_variable,
        variable_leaders=lead_variables,
        **derive_relationship_kwargs,
    )

    def res(inp: pd.DataFrame) -> pd.DataFrame:
        res_h = silicone_infiller(pyam.IamDataFrame(inp)).timeseries()
        # The fact that this is needed suggests there's a bug in silicone
        res_h = res_h.loc[:, inp.dropna(axis="columns", how="all").columns]

        return res_h

    return res


def get_direct_copy_infiller(
    variable: str, copy_from: pd.DataFrame
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Get an infiller which just copies the timeseries from another scenario

    Parameters
    ----------
    variable
        Variable to infill

    copy_from
        Scenario to copy from

    Returns
    -------
    :
        Infiller which can infill data for `variable`
    """

    def infiller(inp: pd.DataFrame) -> pd.DataFrame:
        model_l = inp.index.get_level_values("model").unique()
        if len(model_l) != 1:
            raise AssertionError(model_l)
        model = model_l[0]

        scenario_l = inp.index.get_level_values("scenario").unique()
        if len(scenario_l) != 1:
            raise AssertionError(scenario_l)
        scenario = scenario_l[0]
        mask = copy_from.index.get_level_values("variable") == variable
        res = copy_from[mask].pix.assign(model=model, scenario=scenario)

        return res

    return infiller


def get_direct_scaling_infiller(  # noqa: PLR0913
    leader: str,
    follower: str,
    scaling_factor: float,
    l_0: np.ndarray[float],
    f_0: np.ndarray[float],
    f_unit: str,
    calculation_year: int,
    f_calculation_year: np.ndarray[int],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Get an infiller which just scales one set of emissions to create the next set

    This is basically silicone's constant ratio infiller
    with smarter handling of pre-industrial levels.
    """

    def infiller(inp: pd.DataFrame) -> pd.DataFrame:
        lead_df = inp.loc[pix.isin(variable=[leader])]

        follow_df = (scaling_factor * (lead_df - l_0) + f_0).pix.assign(
            variable=follower, unit=f_unit
        )
        if not np.isclose(follow_df[calculation_year], f_calculation_year).all():
            raise AssertionError

        return follow_df

    return infiller


def infill(
    indf: pd.DataFrame, infillers: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]]
) -> pd.DataFrame | None:
    """
    Infill an emissions scenario using the provided infillers.

    Parameters
    ----------
    indf
        Emissions scenario to infill

    infillers
        Infillers to use

        Each key is the gas the infiller can infill.
        Each value is the function which does the infilling.

    Returns
    -------
    :
        Infilled timeseries.

        If nothing was infilled, `None` is returned
    """
    infilled_l = []
    for variable in infillers:
        for (model, scenario), msdf in indf.groupby(["model", "scenario"]):
            if variable not in msdf.index.get_level_values("variable"):
                infilled_l.append(infillers[variable](msdf))

    if not infilled_l:
        return None

    return pix.concat(infilled_l)


def get_complete(indf: pd.DataFrame, infilled: pd.DataFrame | None) -> pd.DataFrame:
    """
    Get a complete set of timeseries

    This is just a convenience function to help deal with the fact
    that [infill][gcages.cmip7_scenariomip.infilling.infill] can return `None`.

    Parameters
    ----------
    indf
        Input data

    infilled
        Results of infilling using [infill][gcages.cmip7_scenariomip.infilling.infill]

    Returns
    -------
    :
        Complete data i.e. the combination of `indf` and `infilled`
    """
    if infilled is not None:
        complete = pix.concat([indf, infilled])

    else:
        complete = indf

    return complete


def load_cmip7_scenariomip_infilling_db(
    filepath: Path, check_hash: bool = True
) -> pd.DataFrame:
    """
    Load infilling emissions for CMIP7 ScenarioMIP harmonisation.

    Parameters
    ----------
    filepath
        Path from which to load the file

    check_hash
        Check file hash

    Returns
    -------
    :
        Infilled emissions

    Raises
    ------
    AssertionError
        `filepath` does not have the expected hash.

        We expect to be reading the file from
        https://zenodo.org/records/17844114/files/infiling-db_202512021030_202512071232_202511040855_202511040855.csv?download=1
    """
    if check_hash:
        fp_hash = get_file_hash(filepath, algorithm="md5")
        if platform.system() in "Windows":
            fp_hash_exp = "3a55491330c0160a0c0abc011766559a"
        else:
            fp_hash_exp = "3a55491330c0160a0c0abc011766559a"

        if fp_hash != fp_hash_exp:
            msg = (
                f"The md5 hash of {filepath} is {fp_hash}. "
                f"This does not match what we expect {fp_hash_exp=}."
            )
            raise AssertionError(msg)

    res = load_timeseries_csv(
        filepath,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )
    res.columns.name = "year"

    return res


def load_cmip7_scenariomip_ghg_inversions(
    filepath: Path,
) -> pd.DataFrame:
    """
    Load

    Parameters
    ----------
    filepath
        Path from which to load the file

    Returns
    -------
    :
        Green house gases inversion data frame
    """
    res = load_timeseries_csv(
        filepath,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )
    res.columns.name = "year"

    return res


def load_infill_sources(
    cmip7_scenariomip_infilling_leader_emissions_file: Path,
    cmip7_scenariomip_global_historical_emissions_file: Path,
    cmip7_ghg_inversions_file: Path,
) -> InfillingSources:
    """Load all infill files: infilling_db, historical, ghg_inversions."""
    # Still embargoed
    infilling_db = load_cmip7_scenariomip_infilling_db(
        filepath=cmip7_scenariomip_infilling_leader_emissions_file,
        check_hash=False,  # TODO: update when available
    )

    # History
    historical_emissions = load_cmip7_scenariomip_historical_emissions(
        filepath=cmip7_scenariomip_global_historical_emissions_file,
        check_hash=True,
    )

    # CMIP7 GHG inversions
    cmip7_ghg_inversions = load_cmip7_scenariomip_ghg_inversions(
        filepath=cmip7_ghg_inversions_file,
    )

    return InfillingSources(
        infilling_db=infilling_db,
        historical_emissions=historical_emissions,
        cmip7_ghg_inversions=cmip7_ghg_inversions,
    )


def get_pre_industrial_aware_direct_scaling_infiller(
    *,
    historical_emissions: pd.DataFrame,
    cmip7_ghg_inversions_reporting_names: pd.DataFrame,
    scaling_leaders: dict[str, str],
    harmonisation_year: int = 2023,
    pre_industrial_year: int = 1750,
) -> dict[str, Any]:
    """
    Build pre-industrial-aware direct scaling infillers for follower/leader pairs.

    This constructs scaling factors that preserve both pre-industrial baselines and
    harmonisation-year values when scaling follower emissions based on leader emissions.

    The scaling preserves two key properties::

        f_future(l_harmonisation) = f_harmonisation
        f_future(l_pre_industrial) = f_pre_industrial

    with linear interpolation between these anchor points.

    Parameters
    ----------
    historical_emissions
        Historical emissions data with MultiIndex including 'variable' level.
        Must contain harmonisation-year values for all leader/follower variables.

    cmip7_ghg_inversions_reporting_names
        CMIP7 GHG inversion data with pre-industrial (PI) year values.
        Must contain PI-year values for all leader/follower variables.

    scaling_leaders
        Mapping of follower variable names to leader variable names.

    harmonisation_year
        Primary harmonisation reference year

    pre_industrial_year
        Pre-industrial reference year

    Returns
    -------
    :
        Mapping of follower variable into direct scaling infiller callable.

    Raises
    ------
    AssertionError
        If multiple units found for a variable, no valid harmonisation year data,
        or scaling factor computation yields NaN.
    """
    infillers_scaling = {}

    for follower, leader in scaling_leaders.items():
        lead_mask = historical_emissions.index.get_level_values(
            "variable"
        ).str.contains(leader, regex=False)
        lead_df = historical_emissions[lead_mask]
        follow_mask = historical_emissions.index.get_level_values(
            "variable"
        ).str.contains(follower, regex=False)
        follow_df = historical_emissions[follow_mask]

        lead_mask = cmip7_ghg_inversions_reporting_names.index.get_level_values(
            "variable"
        ).str.contains(leader, regex=False)
        lead_cmip7_inverse_df = cmip7_ghg_inversions_reporting_names[lead_mask]
        follow_mask = cmip7_ghg_inversions_reporting_names.index.get_level_values(
            "variable"
        ).str.contains(follower, regex=False)
        follow_cmip7_inverse_df = cmip7_ghg_inversions_reporting_names[follow_mask]

        f_unit = follow_df.index.get_level_values("unit").unique()
        if len(f_unit) != 1:
            msg = f"Multiple units for {follower=}: {f_unit}"
            raise AssertionError(msg)
        f_unit = f_unit[0].replace("-", "")

        l_unit = lead_df.index.get_level_values("unit").unique()
        if len(l_unit) != 1:
            msg = f"Multiple units for {leader=}: {l_unit}"
            raise AssertionError(msg)
        l_unit = l_unit[0].replace("-", "")

        for harmonisation_yr_use in [harmonisation_year, 2021]:
            l_harmonisation_year = lead_df[harmonisation_yr_use].to_numpy().squeeze()

            f_harmonisation_year = follow_df[harmonisation_yr_use].to_numpy().squeeze()

            if not (pd.isnull(l_harmonisation_year) or pd.isnull(f_harmonisation_year)):
                break
        else:
            msg = f"No valid harmonisation year for {follower=}/{leader=}"
            raise AssertionError(msg)

        f_0 = follow_cmip7_inverse_df[pre_industrial_year].to_numpy().squeeze()
        l_0 = lead_cmip7_inverse_df[pre_industrial_year].to_numpy().squeeze()

        scaling_factor = (f_harmonisation_year - f_0) / (l_harmonisation_year - l_0)

        if np.isnan(scaling_factor):
            msg = (
                f"NaN scaling_factor for {follower=}/{leader=}: "
                f"{f_harmonisation_year=} {l_harmonisation_year=} "
                f"{f_0=} {l_0=}"
            )
            raise AssertionError(msg)

        infillers_scaling[follower] = get_direct_scaling_infiller(
            leader=leader,
            follower=follower,
            scaling_factor=scaling_factor,
            l_0=l_0,
            f_0=f_0,
            f_unit=f_unit,
            calculation_year=harmonisation_yr_use,
            f_calculation_year=f_harmonisation_year,
        )

    return infillers_scaling


def create_cmip7_scenariomip_infilled_df(  # noqa: PLR0915
    harmonised_emissions: pd.DataFrame,
    *,
    cmip7_scenariomip_global_historical_emissions_file: Path,
    cmip7_scenariomip_infilling_leader_emissions_file: Path,
    cmip7_ghg_inversions_file: Path,
    ur: openscm_units.unit_registry | None = None,
) -> CMIP7ScenarioMIPInfilledScenarios:
    """
    Create an a infilled df for CMIP7 ScenarioMIP's simple climate model run.

    Parameters
    ----------
    cmip7_scenariomip_infilling_leader_emissions_file
        File containing CMIP7 ScenarioMIP historical emissions.

    Returns
    -------
    :
        Infilled DataFrame
    """
    # if ur is None:
    try:
        import openscm_units

        ur = openscm_units.unit_registry
    except ImportError:
        raise MissingOptionalDependencyError(  # noqa: TRY003
            "convert_unit_like(..., ur=None, ...)", "openscm_units"
        )

    try:
        import pandas_openscm

        pandas_openscm.register_pandas_accessor()
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_silicone_based_infiller", requirement="pandas_openscm"
        ) from exc

    try:
        import silicone.database_crunchers  # type: ignore # silicone has no type hints
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_silicone_based_infiller", requirement="silicone"
        ) from exc

    PI_YEAR = 1750
    HARMONISATION_YEAR = 2023

    infilling_sources = load_infill_sources(
        cmip7_scenariomip_infilling_leader_emissions_file,
        cmip7_scenariomip_global_historical_emissions_file,
        cmip7_ghg_inversions_file,
    )

    infilling_db = infilling_sources.infilling_db
    historical_emissions = infilling_sources.historical_emissions
    cmip7_ghg_inversions = infilling_sources.cmip7_ghg_inversions

    # Check that the infilling database and scenario data are harmonised the same
    assert_harmonised(
        harmonised_emissions,
        history=historical_emissions,
        harmonisation_time=HARMONISATION_YEAR,
        species_aware_cmip7=True,
        ur=ur,
    )
    assert_harmonised(
        infilling_db,
        history=historical_emissions,
        harmonisation_time=HARMONISATION_YEAR,
        species_aware_cmip7=True,
        ur=ur,
    )
    wmo_mask = infilling_db.index.get_level_values("model").str.contains("WMO")
    infilling_wmo = infilling_db[wmo_mask]

    velders_mask = infilling_db.index.get_level_values("model").str.contains("Velders")

    infilling_silicone = infilling_db[~wmo_mask & ~velders_mask]

    # Infill

    ### Very low marker should use F-gas emissions in line with Kigali
    # We get these from [Velders et al., 2022](https://zenodo.org/records/6520707)

    vl_model, vl_scenario = ("REMIND-MAgPIE 3.5-4.11", "SSP1 - Very Low Emissions")

    mask = harmonised_emissions.index.get_level_values("model").str.contains(
        vl_model
    ) & harmonised_emissions.index.get_level_values("scenario").str.contains(
        vl_scenario
    )

    vl_marker = harmonised_emissions[mask]
    unique_var = infilling_silicone.index.get_level_values("variable").unique()

    if not vl_marker.empty:
        lead_vl_marker = "Emissions|CO2|Energy and Industrial Processes"
        infillers_silicone_vl_marker = {}
        for variable in [v for v in unique_var if v != lead_vl_marker]:
            infillers_silicone_vl_marker[variable] = get_silicone_based_infiller(
                infilling_db=infilling_silicone,
                follower_variable=variable,
                lead_variables=[lead_vl_marker],
                silicone_db_cruncher=silicone.database_crunchers.RMSClosest,
            )

        infilled_vl_exception = infill(
            vl_marker,
            infillers_silicone_vl_marker,
        )

    else:
        infilled_vl_exception = None

    complete_vl_exception = get_complete(harmonised_emissions, infilled_vl_exception)

    # Silicone

    lead = "Emissions|CO2|Energy and Industrial Processes"
    infillers_silicone = {}
    for variable in [v for v in unique_var if v != lead]:
        infillers_silicone[variable] = get_silicone_based_infiller(
            infilling_db=infilling_silicone,
            follower_variable=variable,
            lead_variables=[lead],
            silicone_db_cruncher=silicone.database_crunchers.RMSClosest,
        )

    infilled_silicone = infill(
        complete_vl_exception,
        infillers_silicone,
    )
    complete_silicone = get_complete(complete_vl_exception, infilled_silicone)

    # Infill

    infillers_wmo = {}
    unique_var = infilling_wmo.index.get_level_values("variable").unique()
    for wmo_var in unique_var:
        infillers_wmo[wmo_var] = get_direct_copy_infiller(
            variable=wmo_var,
            copy_from=infilling_wmo,
        )

    infilled_wmo = infill(complete_silicone, infillers_wmo)
    complete_wmo = get_complete(complete_silicone, infilled_wmo)

    # Scale timeseries
    #
    # Surprisingly, this is the most mucking around of all.
    # The hard part here is that the scaling needs to be aware
    # of the fact that the pre-industrial value is different for each tiemseries.
    # The naming mucking around also adds to the fun of course.

    to_reporting_names = partial(
        convert_variable_name,
        from_convention=SupportedNamingConventions.GCAGES,
        to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    )

    scaling_leaders_gcages = {
        "Emissions|C3F8": "Emissions|C2F6",
        "Emissions|C4F10": "Emissions|C2F6",
        "Emissions|C5F12": "Emissions|C2F6",
        "Emissions|C7F16": "Emissions|C2F6",
        "Emissions|C8F18": "Emissions|C2F6",
        "Emissions|cC4F8": "Emissions|CF4",
        "Emissions|SO2F2": "Emissions|CF4",
        "Emissions|HFC236fa": "Emissions|HFC245fa",
        "Emissions|HFC152a": "Emissions|HFC4310mee",
        "Emissions|HFC365mfc": "Emissions|HFC134a",
        "Emissions|CH2Cl2": "Emissions|HFC134a",
        "Emissions|CHCl3": "Emissions|C2F6",
        "Emissions|NF3": "Emissions|SF6",
    }
    scaling_leaders = {
        to_reporting_names(k): to_reporting_names(v)
        for k, v in scaling_leaders_gcages.items()
    }

    cmip7_ghg_inversions_reporting_names = update_index_levels_func(
        cmip7_ghg_inversions, {"variable": to_reporting_names}
    )

    infillers_scaling = get_pre_industrial_aware_direct_scaling_infiller(
        historical_emissions=historical_emissions,
        cmip7_ghg_inversions_reporting_names=cmip7_ghg_inversions_reporting_names,
        scaling_leaders=scaling_leaders,
        harmonisation_year=HARMONISATION_YEAR,
        pre_industrial_year=PI_YEAR,
    )

    infilled_scaling = infill(complete_wmo, infillers_scaling)
    complete = get_complete(complete_wmo, infilled_scaling)

    ## Check completeness
    assert_all_groups_are_complete(complete, complete_index_reporting_names)

    infilled = CMIP7ScenarioMIPInfilledScenarios()
    for ids, df in (
        ("silicone", infilled_silicone),
        ("wmo", infilled_wmo),
        ("scaled", infilled_scaling),
        ("complete", complete),
    ):
        if df is not None:
            years = [c for c in df.columns if pd.api.types.is_numeric_dtype([c])]
            other_cols = [c for c in df.columns if c not in years]
            infilled._add(ids, df[other_cols + sorted(years)])

    return infilled
