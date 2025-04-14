"""
Infilling part of the AR6 workflow
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import pandas as pd
from attrs import define
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv
from pandas_openscm.parallelisation import ParallelOpConfig, apply_op_parallel_progress

from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_index_levels,
    assert_index_is_multiindex,
    assert_only_working_on_variable_unit_variations,
)
from gcages.completeness import assert_all_groups_are_complete
from gcages.exceptions import MissingOptionalDependencyError
from gcages.harmonisation import assert_harmonised
from gcages.hashing import get_file_hash
from gcages.renaming import SupportedNamingConventions, convert_variable_name

if TYPE_CHECKING:
    import silicone  # type: ignore


@functools.cache
def load_massaged_ar6_infilling_db(filepath: Path, cfcs: bool) -> pd.DataFrame:
    """
    Load the infilling database that was used in AR6

    The data is massaged to what is expected by our infilling,
    it isn't the raw data (at least not raw metadata).

    Parameters
    ----------
    filepath
        Filepath from which to load the infilling database

    cfcs
        If `True`, we expect to load the database for infilling CFCs

    Returns
    -------
    :
        Infilling database used in AR6

    Raises
    ------
    AssertionError
        `filepath` points to a file that does not have the expected hash
    """
    fp_hash = get_file_hash(filepath, algorithm="sha256")
    if cfcs:
        fp_hash_exp = "9ad0550c671701622ec2b2e7ba2b6c38d58f83507938ab0f5aa8b1a35d26c015"

    else:
        fp_hash_exp = "4ef7aabb18c35fdf857145adfbfffbbcfd6667b5551cc0b7a182e682f03b5843"

    if fp_hash != fp_hash_exp:
        msg = (
            f"The sha256 hash of {filepath} is {fp_hash}. "
            f"This does not match what we expect ({fp_hash_exp=})."
        )
        raise AssertionError(msg)

    res = load_timeseries_csv(
        filepath,
        lower_column_names=True,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_column_type=int,
    )

    if cfcs:
        # Not sure why this was like this, anyway
        res = update_index_levels_func(
            res,
            {
                # Another naming convention?
                # Probably have to check RCMIP...
                "variable": lambda x: x.replace("|HFC|", "|")
                .replace("|PFC|", "|")
                .replace("Energy and Industrial Processes", "Fossil"),
            },
        )

    else:
        res = update_index_levels_func(
            res,
            {
                "variable": lambda x: x.replace(
                    "AR6 climate diagnostics|Harmonized|", ""
                )
            },
        )
        res = res.loc[
            ~res.index.get_level_values("variable").isin(
                # Variables that aren't actually used
                ["Emissions|F-Gases", "Emissions|HFC", "Emissions|PFC"]
            )
        ]
        res = update_index_levels_func(
            res,
            {
                "variable": partial(
                    convert_variable_name,
                    from_convention=SupportedNamingConventions.IAMC,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            },
        )

    return res


@functools.cache
def get_ar6_full_historical_emissions(filepath: Path) -> pd.DataFrame:
    """
    Get the full AR6 historical emissions

    Parameters
    ----------
    filepath
        Filepath from which to load the emissions

    Returns
    -------
    :
        Historical emissions as used in AR6

    Raises
    ------
    AssertionError
        `filepath` points to a file that does not have the expected hash
    """
    try:
        from pandas_indexing.core import assignlevel, projectlevel
        from pandas_indexing.selectors import isin, ismatch
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_ar6_full_historical_emissions", requirement="pandas_indexing"
        ) from exc

    fp_hash = get_file_hash(filepath, algorithm="sha256")
    fp_hash_exp = "9ad0550c671701622ec2b2e7ba2b6c38d58f83507938ab0f5aa8b1a35d26c015"

    if fp_hash != fp_hash_exp:
        msg = (
            f"The sha256 hash of {filepath} is {fp_hash}. "
            f"This does not match what we expect ({fp_hash_exp=})."
        )
        raise AssertionError(msg)

    raw = load_timeseries_csv(
        filepath,
        lower_column_names=True,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_column_type=int,
    )

    history = raw.loc[
        isin(scenario="ssp245") & ~ismatch(variable="**CO2"), :2015
    ].reset_index(["model", "scenario"], drop=True)

    history = assignlevel(
        history,
        variable=projectlevel(history.index, "variable").map(
            lambda x: convert_variable_name(
                x,
                from_convention=SupportedNamingConventions.AR6_CFC_INFILLING_DB,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        ),
        # Not strictly necessary, but makes life easier
        unit=projectlevel(history.index, "unit").map(
            lambda x: x.replace("HFC4310mee/yr", "HFC4310/yr").replace(
                "NO2 / yr", "NO2/yr"
            )
        ),
    )
    # Not sure why this happened, but here we are
    history.loc[ismatch(variable="**HFC245fa"), :] *= 0.0  # type: ignore # pix typing not playing nice with pandas-stubs

    return history


def infill_scenario(
    indf: pd.DataFrame,
    infillers: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
) -> pd.DataFrame:
    """
    Infill a single scenario

    Parameters
    ----------
    indf
        Scenario to harmonise

    infillers
        Functions to use for infilling each variable.

        The keys define the variable that can be infilled.
        The variables define the function which,
        given inputs with the expected lead variables,
        returns the infilled time series.

    Returns
    -------
    :
        Infilled scenario
    """
    try:
        from pandas_indexing.core import concat, uniquelevel
        from pandas_indexing.selectors import isin
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "infill_scenario", requirement="pandas_indexing"
        ) from exc

    assert_only_working_on_variable_unit_variations(indf)

    indf_variables = uniquelevel(indf, "variable")

    inferred_l = []
    # Don't repeat this logic again, it is super confusing
    # and should be done in pre-processing instead.
    if "Emissions|CO2" in indf_variables:
        for component_missing, component_other in (
            ("Emissions|CO2|Fossil", "Emissions|CO2|Biosphere"),
            ("Emissions|CO2|Biosphere", "Emissions|CO2|Fossil"),
        ):
            if (
                component_missing not in indf_variables
                and component_other in indf_variables
            ):
                inferred = (
                    indf.loc[isin(variable=["Emissions|CO2"])]
                    .subtract(
                        indf.loc[isin(variable=[component_other])].reset_index(
                            "variable", drop=True
                        ),
                        axis="rows",  # type: ignore # pandas-stubs confused
                    )
                    .pix.assign(variable=component_missing)
                )
                inferred_l.append(inferred)

    if inferred_l:
        inferred = concat(inferred_l)
        inferred_variables = inferred.pix.unique("variable")
    else:
        inferred = None
        inferred_variables = []

    to_infill_silicone = [
        v for v in infillers if v not in indf_variables and v not in inferred_variables
    ]

    infilled_silicone_l = []
    for v_to_infill in to_infill_silicone:
        infiller = infillers[v_to_infill]
        tmp = infiller(indf)
        # The fact that this is needed suggests there's a bug in silicone
        tmp = tmp.loc[:, indf.columns]  # type: ignore # pandas-stubs being stupid
        infilled_silicone_l.append(tmp)

    # Also add zeros for HFC245fa.
    # The fact that this is effectively hidden is an illustration
    # of the issues with how our stack is set up.
    tmp = (tmp * 0.0).pix.assign(variable="Emissions|HFC245fa", unit="kt HFC245fa/yr")
    infilled_silicone_l.append(tmp)

    infilled_silicone = concat(infilled_silicone_l).sort_index(axis="columns")
    infilled_silicone_vars = uniquelevel(infilled_silicone, "variable")

    # If we started with total CO2 then infilled fossil CO2,
    # we preserve harmonisation and the total by overwriting AFOLU CO2.
    if (
        "Emissions|CO2" in indf_variables
        and "Emissions|CO2|Fossil" in infilled_silicone_vars
    ):
        infilled_silicone = concat(
            [
                infilled_silicone.loc[~isin(variable=["Emissions|CO2|Biosphere"])],
                (
                    indf.loc[isin(variable=["Emissions|CO2"])]
                    .subtract(
                        infilled_silicone.loc[
                            isin(variable=["Emissions|CO2|Fossil"])
                        ].reset_index("variable", drop=True),
                        axis="rows",  # type: ignore # pandas-stubs confused
                    )
                    .pix.assign(variable="Emissions|CO2|Biosphere")
                ),
            ]
        )

    if inferred is not None:
        infilled: pd.DataFrame = concat([inferred, infilled_silicone])
    else:
        infilled = infilled_silicone

    return infilled


@functools.cache
def get_ar6_infiller(  # type: ignore # silicone has no type hints
    follower: str,
    lead: tuple[str, ...],
    db_file: Path,
    cruncher: silicone.base._DatabaseCruncher,
    cfcs: bool,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Get an infiller following the logic used in AR6

    Parameters
    ----------
    follower
        Variable to infill

    lead
        Lead variable(s) for the infilling

    db_file
        File from which to load the infilling database

    cruncher
        Cruncher to use for deriving the relationship between `follower` and `lead`

    cfcs
        Should we load the CFC infilling database?

    Returns
    -------
    :
        Infilled timeseries
    """
    try:
        import pyam  # type: ignore # pyam not typed
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "get_ar6_infiller", requirement="pyam"
        ) from exc

    db = load_massaged_ar6_infilling_db(db_file, cfcs=cfcs)

    infiller = cruncher(pyam.IamDataFrame(db)).derive_relationship(
        variable_follower=follower,
        variable_leaders=list(lead),
    )

    def res_func(inp: pd.DataFrame) -> pd.DataFrame:
        return cast(pd.DataFrame, infiller(pyam.IamDataFrame(inp)).timeseries())

    return res_func


def do_ar6_like_infilling(  # type: ignore # noqa: PLR0913 # silicone has no type hints
    indf: pd.DataFrame,
    follower: str,
    lead_options: tuple[tuple[str, ...], ...],
    db_file: Path,
    cruncher: silicone.base._DatabaseCruncher,
    cfcs: bool,
) -> pd.DataFrame:
    """
    Do infilling like it was done in AR6

    Parameters
    ----------
    indf
        Emissions scenario to infill

    follower
        Variable to infill

    lead_options
        Options to try as the lead variable

    db_file
        File from which to load the infilling database

    cruncher
        Cruncher to use for deriving the relationship between `follower` and `lead`

    cfcs
        Should we load the CFC infilling database?

    Returns
    -------
    :
        Infilled timeseries
    """
    indf_variables = indf.index.get_level_values("variable")
    for leads in lead_options:
        if all(v in indf_variables for v in leads):
            # This function is split out to allow for caching
            infiller = get_ar6_infiller(
                follower=follower,
                lead=leads,
                db_file=db_file,
                cruncher=cruncher,
                cfcs=cfcs,
            )

            res = infiller(indf)
            break

    else:
        raise NotImplementedError

    return res


@define
class AR6Infiller:
    """
    Infiller that follows the same logic as was used in AR6

    If you want exactly the same behaviour as in AR6,
    initialise using [`from_ar6_config`][(c)]
    """

    infillers: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]]
    """
    Functions to use for infilling each variable.

    The keys define the variable that can be infilled.
    The variables define the function which,
    given inputs with the expected lead variables,
    returns the infilled timeseries.
    """

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    historical_emissions: pd.DataFrame | None = None
    """
    Historical emissions used for harmonisation

    Only required if `run_checks` is `True` to check
    that the infilled data is also harmonised.
    """

    harmonisation_year: int | None = None
    """
    Year in which the data was harmonised

    Only required if `run_checks` is `True` to check
    that the infilled data is also harmonised.
    """

    progress: bool = True
    """
    Should progress bars be shown for each operation?
    """

    n_processes: int | None = None  # better off in serial with silicone
    """
    Number of processes to use for parallel processing.

    Set to `None` to process in serial.
    """

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Infill

        Parameters
        ----------
        in_emissions
            Emissions to infill

        Returns
        -------
        :
            Infilled emissions
        """
        if self.run_checks:
            assert_index_is_multiindex(in_emissions)
            assert_data_is_all_numeric(in_emissions)
            # Needed for parallelisation
            assert_has_index_levels(
                in_emissions, ["variable", "unit", "model", "scenario"]
            )

        # Strip off any prefixes that might be there
        to_infill = in_emissions.pix.assign(
            variable=in_emissions.index.get_level_values("variable").map(
                lambda x: x.replace("AR6 climate diagnostics|Harmonized|", "")
            )
        ).sort_index(axis="columns")

        # This is not the most efficient parallelisation.
        # It would be better to group by variable,
        # then have an `infill_variable` function,
        # but this would require a bit more thinking
        # to actually achieve.
        infilled = pd.concat(
            [
                v.reorder_levels(to_infill.index.names)
                for v in apply_op_parallel_progress(
                    func_to_call=infill_scenario,
                    iterable_input=(
                        gdf for _, gdf in to_infill.groupby(["model", "scenario"])
                    ),
                    parallel_op_config=ParallelOpConfig.from_user_facing(
                        progress=self.progress,
                        max_workers=self.n_processes,
                        progress_results_kwargs=dict(desc="Scenarios to infill"),
                    ),
                    infillers=self.infillers,
                )
            ]
        )

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
                history=self.historical_emissions,
                harmonisation_time=self.harmonisation_year,
                rounding=5,  # level of data storage in historical data often
            )
            assert_all_groups_are_complete(
                # The combo of the input and infilled should be complete
                pd.concat(
                    [in_emissions, infilled.reorder_levels(in_emissions.index.names)]
                ),
                complete_index=self.historical_emissions.index.droplevel("unit"),
            )

        return infilled

    @classmethod
    def from_ar6_config(  # noqa: PLR0913
        cls,
        ar6_infilling_db_file: Path,
        ar6_infilling_db_cfcs_file: Path,
        variables_to_infill: Iterable[str] | None = None,
        run_checks: bool = True,
        historical_emissions: pd.DataFrame | None = None,
        harmonisation_year: int | None = None,
        progress: bool = True,
        n_processes: int | None = None,  # better off in serial with silicone
    ) -> AR6Infiller:
        """
        Initialise from the config used in AR6

        Parameters
        ----------
        ar6_infilling_db_file
            File containing the AR6 infilling database

            This is for all emissions except CFCs.

        ar6_infilling_db_cfcs_file
            File containing the AR6 infilling database for CFCs

        variables_to_infill
            Variables to infill.

            If not supplied, we use the default set from AR6.

        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        historical_emissions
            Historical emissions used for harmonisation

            Only required if `run_checks` is `True` to check
            that the infilled data is also harmonised.

        harmonisation_year
            Year in which the data was harmonised

            Only required if `run_checks` is `True` to check
            that the infilled data is also harmonised.

        progress
            Should a progress bar be shown for each operation?

        n_processes
            Number of processes to use for parallel processing.

            Set to `None` to process in serial.

        Returns
        -------
        :
            Initialised harmoniser
        """
        try:
            import silicone.database_crunchers  # type: ignore # silicone has no type hints
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "get_ar6_infiller", requirement="silicone"
            ) from exc

        VARS_DB_CRUNCHERS = {
            "Emissions|BC": (
                False,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|CH4": (
                False,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|CO2|Biosphere": (
                False,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|CO2|Fossil": (
                False,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|CO": (
                False,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|N2O": (
                False,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|NH3": (
                False,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|NOx": (
                False,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|OC": (
                False,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|SOx": (
                False,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|NMVOC": (
                False,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|HFC134a": (
                False,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC143a": (
                False,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC227ea": (
                False,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC23": (
                False,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC32": (
                False,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC4310mee": (
                False,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC125": (
                False,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|SF6": (
                False,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CF4": (
                False,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|C2F6": (
                False,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|C6F14": (
                False,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CCl4": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CFC11": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CFC113": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CFC114": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CFC115": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CFC12": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CH2Cl2": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CH3Br": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CH3CCl3": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CH3Cl": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CHCl3": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HCFC141b": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HCFC142b": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HCFC22": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC152a": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC236fa": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC365mfc": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|Halon1202": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|Halon1211": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|Halon1301": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|Halon2402": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|NF3": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|C3F8": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|C4F10": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|C5F12": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|C7F16": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|C8F18": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|cC4F8": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|SO2F2": (
                True,
                silicone.database_crunchers.RMSClosest,
            ),
        }
        """
        Definition of our default set of variables to infill and how to infill them

        Each key is a variable we can infill.
        Each value is a tuple with the following:

        - `False` if we should use the 'full' infilling database,
          `True` if we should use the database
          that has information about CFCs and other species
          not typically modelled by IAMs
        - The database cruncher from silicone to use for infilling the variable
        """

        if variables_to_infill is None:
            # Technically, having to infill some of these
            # would have caused a scenario to fail vetting,
            # but we could at least in theory infill all of them.
            variables_to_infill = tuple(VARS_DB_CRUNCHERS.keys())

        lead_options = (
            ("Emissions|CO2",),
            ("Emissions|CO2|Fossil",),
        )

        infillers = {}

        for v_infill in variables_to_infill:
            cfcs, cruncher = VARS_DB_CRUNCHERS[v_infill]

            infillers[v_infill] = partial(
                do_ar6_like_infilling,
                follower=v_infill,
                lead_options=lead_options,
                db_file=ar6_infilling_db_file
                if not cfcs
                else ar6_infilling_db_cfcs_file,
                cruncher=cruncher,
                cfcs=cfcs,
            )

        # CO2 Energy and Industrial special case
        infillers["Emissions|CO2|Fossil"] = partial(
            do_ar6_like_infilling,
            follower="Emissions|CO2|Fossil",
            lead_options=(("Emissions|CO2",),),
            db_file=ar6_infilling_db_file,
            cruncher=VARS_DB_CRUNCHERS["Emissions|CO2|Fossil"][1],
            cfcs=False,
        )

        return cls(
            infillers=infillers,
            run_checks=run_checks,
            historical_emissions=historical_emissions,
            harmonisation_year=harmonisation_year,
            progress=progress,
            n_processes=n_processes,
        )
