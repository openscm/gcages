"""
Infilling configuration and related things for the CMIP7 ScenarioMIP workflow
"""

from __future__ import annotations

import platform
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from attrs import define
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv

from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_index_levels,
    assert_index_is_multiindex,
)
from gcages.cmip7_scenariomip.harmonisation import (
    load_cmip7_scenariomip_historical_emissions,
)
from gcages.completeness import assert_all_groups_are_complete
from gcages.exceptions import MissingOptionalDependencyError
from gcages.harmonisation.common import assert_harmonised
from gcages.hashing import get_file_hash
from gcages.renaming import SupportedNamingConventions, convert_variable_name

if TYPE_CHECKING:
    import silicone  # type: ignore[import-untyped]
    from pint import UnitRegistry


COMPLETE_EMISSIONS_INPUT_VARIABLES_GCAGES = [
    "Emissions|CO2|Biosphere",
    "Emissions|CO2|Fossil",
    "Emissions|BC",
    "Emissions|CH4",
    "Emissions|CO",
    "Emissions|N2O",
    "Emissions|NH3",
    "Emissions|NMVOC",
    "Emissions|NOx",
    "Emissions|OC",
    "Emissions|SOx",
    "Emissions|C2F6",
    "Emissions|C6F14",
    "Emissions|CF4",
    "Emissions|SF6",
    "Emissions|HFC125",
    "Emissions|HFC134a",
    "Emissions|HFC143a",
    "Emissions|HFC227ea",
    "Emissions|HFC23",
    "Emissions|HFC245fa",
    "Emissions|HFC32",
    "Emissions|HFC4310mee",
    "Emissions|CCl4",
    "Emissions|CFC11",
    "Emissions|CFC113",
    "Emissions|CFC114",
    "Emissions|CFC115",
    "Emissions|CFC12",
    "Emissions|CH3CCl3",
    "Emissions|HCFC141b",
    "Emissions|HCFC142b",
    "Emissions|HCFC22",
    "Emissions|Halon1202",
    "Emissions|Halon1211",
    "Emissions|Halon1301",
    "Emissions|Halon2402",
    "Emissions|C3F8",
    "Emissions|C4F10",
    "Emissions|C5F12",
    "Emissions|C7F16",
    "Emissions|C8F18",
    "Emissions|cC4F8",
    "Emissions|SO2F2",
    "Emissions|HFC236fa",
    "Emissions|HFC152a",
    "Emissions|HFC365mfc",
    "Emissions|CH2Cl2",
    "Emissions|CHCl3",
    "Emissions|CH3Br",
    "Emissions|CH3Cl",
    "Emissions|NF3",
]
"""
Complete set of input emissions using gcages' naming
"""


complete_index_gcages_names = pd.MultiIndex.from_product(
    [COMPLETE_EMISSIONS_INPUT_VARIABLES_GCAGES, ["World"]],
    names=["variable", "region"],
)
"""
Complete index using gcages' names
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
        import pyam  # type: ignore
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

        return cast(pd.DataFrame, res_h)

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
        try:
            from pandas_indexing.selectors import isin as pix_isin
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "get_direct_copy_infiller", requirement="pandas_indexing"
            ) from exc

        model_l = inp.index.get_level_values("model").unique()
        if len(model_l) != 1:
            raise AssertionError(model_l)
        model = model_l[0]

        scenario_l = inp.index.get_level_values("scenario").unique()
        if len(scenario_l) != 1:
            raise AssertionError(scenario_l)
        scenario = scenario_l[0]

        res = copy_from.loc[pix_isin(variable=variable)].pix.assign(
            model=model, scenario=scenario
        )

        return cast(pd.DataFrame, res)

    return infiller


def get_direct_scaling_infiller(  # noqa: PLR0913
    leader: str,
    follower: str,
    scaling_factor: float,
    l_0: float,
    f_0: float,
    f_unit: str,
    calculation_year: int,
    f_calculation_year: int,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Get an infiller which just scales one set of emissions to create the next set

    This is basically silicone's constant ratio infiller
    with smarter handling of pre-industrial levels.
    """

    def infiller(inp: pd.DataFrame) -> pd.DataFrame:
        mask = inp.index.get_level_values("variable").str.contains(leader, regex=False)
        lead_df = inp[mask]

        follow_df = (scaling_factor * (lead_df - l_0) + f_0).pix.assign(
            variable=follower, unit=f_unit
        )
        if not np.isclose(follow_df[calculation_year], f_calculation_year).all():
            raise AssertionError

        return cast(pd.DataFrame, follow_df)

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
    try:
        from pandas_indexing.core import concat
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "infill", requirement="pandas_indexing"
        ) from exc

    infilled_l = []
    for variable in infillers:
        for (model, scenario), msdf in indf.groupby(["model", "scenario"]):
            if variable not in msdf.index.get_level_values("variable"):
                infilled_l.append(infillers[variable](msdf))

    if not infilled_l:
        return None

    return concat(infilled_l)


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
    try:
        from pandas_indexing.core import concat
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "infill", requirement="pandas_indexing"
        ) from exc

    if infilled is not None:
        complete = concat([indf, infilled])

    else:
        complete = indf

    return complete


def load_cmip7_scenariomip_infilling_db(
    filepath: Path, check_hash: bool = True
) -> pd.DataFrame:
    """
    Load infilling database for CMIP7 ScenarioMIP harmonisation.

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
        out_columns_name="year",
    )

    return res


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
        f_unit_str = str(f_unit[0]).replace("-", "")

        l_unit = lead_df.index.get_level_values("unit").unique()
        if len(l_unit) != 1:
            msg = f"Multiple units for {leader=}: {l_unit}"
            raise AssertionError(msg)
        l_unit = l_unit[0].replace("-", "")

        for harmonisation_yr_use in [harmonisation_year, 2021]:
            l_harmonisation_year = (
                lead_df[harmonisation_yr_use].to_numpy().squeeze().item()
            )

            f_harmonisation_year = (
                follow_df[harmonisation_yr_use].to_numpy().squeeze().item()
            )

            if not (pd.isnull(l_harmonisation_year) or pd.isnull(f_harmonisation_year)):
                break
        else:
            msg = f"No valid harmonisation year for {follower=}/{leader=}"
            raise AssertionError(msg)

        f_0 = follow_cmip7_inverse_df[pre_industrial_year].to_numpy().squeeze().item()
        l_0 = lead_cmip7_inverse_df[pre_industrial_year].to_numpy().squeeze().item()

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
            f_unit=f_unit_str,
            calculation_year=harmonisation_yr_use,
            f_calculation_year=f_harmonisation_year,
        )

    return infillers_scaling


@define
class CMIP7ScenarioMIPInfiller:
    """
    Infiller that follows the same logic as was used in CMIP7 ScenarioMIP

    If you want exactly the same behaviour as in CMIP7 ScenarioMIP,
    initialise using [`from_cmip7_scenariomip_config`][(c)]
    """

    infilling_db: pd.DataFrame
    """
    Infilling leaders data base for each variable.
    """

    cmip7_ghg_inversions: pd.DataFrame
    """
    Green house gasses inversion data frame.
    """

    historical_emissions: pd.DataFrame
    """
    Historical emissions used for harmonisation
    """
    harmonisation_year: int | None = 2023
    """
    Year in which the data was harmonised
    """
    pre_industrial_year: int | None = 1750
    """
    Pre-Industrial year
    """
    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    ur: UnitRegistry | None = (None,)
    """
    UnitRegistry
    """

    cmip7_scenariomip_output: bool = False
    """
    Output equivalent to CMIP7 ScenarioMIP
    """

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0915, PLR0912
        """
        Create an a infilled df for CMIP7 ScenarioMIP's simple climate model run.

        Parameters
        ----------
        in_emissions
            Emissions to infill

        Returns
        -------
        :
            Infilled emissions DataFrame
        """
        if self.ur is None:
            try:
                import openscm_units

                self.ur = openscm_units.unit_registry
            except ImportError as exc:
                raise MissingOptionalDependencyError(
                    "openscm_units",
                    requirement="openscm_units",
                ) from exc

        try:
            import silicone.database_crunchers  # type: ignore # silicone has no type hints
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "get_silicone_based_infiller", requirement="silicone"
            ) from exc

        # Use gcages naming convention.
        in_emissions = update_index_levels_func(
            in_emissions,
            {
                "variable": lambda x: convert_variable_name(
                    x,
                    from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            },
            copy=False,
        )

        if self.run_checks:
            assert_index_is_multiindex(in_emissions)
            assert_data_is_all_numeric(in_emissions)
            assert_has_index_levels(
                in_emissions, ["variable", "unit", "model", "scenario"]
            )
            # Check that the infilling database and
            # scenario data are harmonised the same
            history = self.historical_emissions.reset_index(
                level=[
                    lvl
                    for lvl in ["model", "scenario"]
                    if lvl in self.historical_emissions.index.names
                ],
                drop=True,
            )
            assert_harmonised(
                in_emissions,
                history=history,
                harmonisation_time=self.harmonisation_year,
            )

        infilling_wmo = self.infilling_db[
            self.infilling_db.index.get_level_values("model").str.contains("WMO")
        ]

        infilling_silicone = self.infilling_db[
            ~self.infilling_db.index.get_level_values("model").str.contains("WMO")
            & ~self.infilling_db.index.get_level_values("model").str.contains("Velders")
        ]

        # Infill

        # TODO: split this out somehow
        ### Very low marker should use F-gas emissions in line with Kigali
        # We get these from [Velders et al., 2022](https://zenodo.org/records/6520707)

        vl_model, vl_scenario = ("REMIND-MAgPIE 3.5-4.11", "SSP1 - Very Low Emissions")

        mask = in_emissions.index.get_level_values("model").str.contains(
            vl_model
        ) & in_emissions.index.get_level_values("scenario").str.contains(vl_scenario)

        vl_marker = in_emissions[mask]
        unique_var = infilling_silicone.index.get_level_values("variable").unique()
        if not vl_marker.empty:
            lead_vl_marker = "Emissions|CO2|Fossil"
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

        complete_vl_exception = get_complete(in_emissions, infilled_vl_exception)

        # Silicone
        lead = "Emissions|CO2|Fossil"
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

        scaling_leaders = {
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

        infillers_scaling = get_pre_industrial_aware_direct_scaling_infiller(
            historical_emissions=self.historical_emissions,
            cmip7_ghg_inversions_reporting_names=self.cmip7_ghg_inversions,
            scaling_leaders=scaling_leaders,
            harmonisation_year=self.harmonisation_year,
            pre_industrial_year=self.pre_industrial_year,
        )

        infilled_scaling = infill(complete_wmo, infillers_scaling)
        infilled = get_complete(complete_wmo, infilled_scaling)

        if self.run_checks:
            pd.testing.assert_index_equal(infilled.columns, in_emissions.columns)

            if self.historical_emissions is None:
                msg = "`self.historical_emissions` must be set to check the infilling"
                raise AssertionError(msg)

            if self.harmonisation_year is None:
                msg = "`self.harmonisation_year` must be set to check the infilling"
                raise AssertionError(msg)

            assert_harmonised(
                infilled,
                history=history,
                harmonisation_time=self.harmonisation_year,
                rounding=5,  # level of data storage in historical data often
            )
            ## Check completeness
            assert_all_groups_are_complete(infilled, complete_index_gcages_names)

        if self.cmip7_scenariomip_output:
            # Use revert to cmip7 ScenatioMIP naming convention.
            infilled = update_index_levels_func(
                infilled,
                {
                    "variable": lambda x: convert_variable_name(
                        x,
                        from_convention=SupportedNamingConventions.GCAGES,
                        to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                    )
                },
                copy=False,
            )

        return infilled

    @classmethod
    def from_cmip7_scenariomip_config(
        cls,
        cmip7_scenariomip_infilling_leader_emissions_file: Path,
        cmip7_ghg_inversions_file: Path,
        cmip7_scenariomip_global_historical_emissions_file: Path,
        ur: UnitRegistry | None = None,
        run_checks: bool = True,
    ) -> CMIP7ScenarioMIPInfiller:
        """
        Initialise from the config used in AR6

        Parameters
        ----------
        cmip7_scenariomip_infilling_leader_emissions_file
            File containing the infilling leaders database

            This is for all emissions except GHGs.

        cmip7_ghg_inversions_file
            File containing the infilling database for GHGs inversions

        cmip7_scenariomip_global_historical_emissions_file
            File containing the historical emissions used for harmonisation

        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        Returns
        -------
        :
            Initialised CMIP7ScenarioMIPInfiller
        """
        # Hardcode as we are matching CMIP7 ScenarioMIP exactly.
        # Users can copy and modify themselves if they wish
        # (or we can introduce a lower layer if lots of users want it)
        PI_YEAR = 1750
        HARMONISATION_YEAR = 2023

        if ur is None:
            try:
                import openscm_units

                ur = openscm_units.unit_registry
            except ImportError as exc:
                raise MissingOptionalDependencyError(
                    "openscm_units",
                    requirement="openscm_units",
                ) from exc

        # Still embargoed
        infilling_db = load_cmip7_scenariomip_infilling_db(
            filepath=cmip7_scenariomip_infilling_leader_emissions_file,
            check_hash=False,  # TODO: update when available
        )

        # CMIP7 GHG inversions
        cmip7_ghg_inversions = load_cmip7_scenariomip_ghg_inversions(
            filepath=cmip7_ghg_inversions_file,
        )
        # History
        historical_emissions = load_cmip7_scenariomip_historical_emissions(
            filepath=cmip7_scenariomip_global_historical_emissions_file,
            check_hash=True,
        )

        # Use gcages naming convention.
        infilling_db = update_index_levels_func(
            infilling_db,
            {
                "variable": lambda x: convert_variable_name(
                    x,
                    from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            },
            copy=False,
        )
        cmip7_ghg_inversions = update_index_levels_func(
            cmip7_ghg_inversions,
            {
                "variable": lambda x: convert_variable_name(
                    x,
                    from_convention=SupportedNamingConventions.OPENSCM_RUNNER,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            },
            copy=False,
        )
        historical_emissions = update_index_levels_func(
            historical_emissions,
            {
                "variable": lambda x: convert_variable_name(
                    x,
                    from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            },
            copy=False,
        )

        if run_checks:
            assert_harmonised(
                infilling_db,
                history=historical_emissions.reset_index(
                    level=[
                        lvl
                        for lvl in ["model", "scenario"]
                        if lvl in historical_emissions.index.names
                    ],
                    drop=True,
                ),
                harmonisation_time=HARMONISATION_YEAR,
                history_unit_level="unit",
                ur=ur,
            )
        # TODO not sure here
        cmip7_scenariomip_output = True

        return cls(
            infilling_db=infilling_db,
            historical_emissions=historical_emissions,
            cmip7_ghg_inversions=cmip7_ghg_inversions,
            harmonisation_year=HARMONISATION_YEAR,
            pre_industrial_year=PI_YEAR,
            run_checks=run_checks,
            ur=ur,
            cmip7_scenariomip_output=cmip7_scenariomip_output,
        )
