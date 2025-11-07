"""
Infilling using [silicone][]
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

import pandas as pd
import silicone
import silicone.database_crunchers
from attrs import Factory, define

from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_index_levels,
    assert_index_is_multiindex,
)
from gcages.completeness import assert_all_groups_are_complete
from gcages.exceptions import MissingOptionalDependencyError
from gcages.harmonisation import assert_harmonised

if TYPE_CHECKING:
    import silicone
    import silicone.database_crunchers


@define
class SiliconeBasedInfillingConfig:
    """
    Configuration of an infilling source for [silicone][]-based infilling
    """

    db: Callable[[], pd.DataFrame]
    """
    A callable which returns the database to use for the infilling when called.

    Making this a callable allows for lazy loading and caching.
    """

    lead: str | tuple[str, ...]
    """The lead gas to use for the infilling"""

    cruncher: silicone.database_crunchers.base._DatabaseCruncher
    """Cruncher to use for the infilling"""


@define
class SiliconeBasedInfiller:
    """
    Infiller that uses [silicone][]
    """

    infilling_configuration: Mapping[str, SiliconeBasedInfillingConfig]
    """
    Configuration to use for infilling

    Each key is a variable that can be infilled.
    Each value is the configuration to use for infilling the variable.
    """

    run_checks: bool = True
    """
    Should checks of the input and output data be performed?

    If this is turned off, things are faster,
    but error messages are much less clear if things go wrong.
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

    infillers: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] = Factory(dict)
    """
    Callables to use for infilling

    As these callables are derived, they are stored here.
    This is essentially a cache,
    users should not need to modify this attribute themselves.
    """

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        # check input
        if self.run_checks:
            assert_index_is_multiindex(in_emissions)
            assert_data_is_all_numeric(in_emissions)
            assert_has_index_levels(
                in_emissions,
                [
                    self.variable_level,
                    self.region_level,
                    self.unit_level,
                    # Needed for parallelisation
                    *self.scenario_group_levels,
                ],
            )

        # Get missing index from infilling_configuration and scenario groups
        # Group missing index by variable
        # For scenarios that need that variable infilled
        # Make sure required lead variable is there
        # Infill
        # Add to list
        # Repeat
        # Combine
        # Done
        import itertools

        assert False, "Think about region level"
        complete_idx = pd.MultiIndex.from_tuples(
            [
                tuple(
                    value
                    for level_nested_element in level_nested
                    for value in level_nested_element
                )
                for level_nested in tuple(
                    itertools.product(
                        self.infilling_configuration.keys(),
                        in_emissions.index.droplevel(
                            in_emissions.index.names.difference(
                                self.scenario_group_levels
                            )
                        ).values,
                    )
                )
            ],
            names=["variable", *self.scenario_group_levels],
        )
        missing_levels = get_missing_levels(
            in_emissions.index,
            complete_index=complete_idx,
            unit_col=self.unit_level,
        )
        explode

        infilled_l = []
        for variable in self.infilling_configuration:
            if variable not in self.infillers:
                self.infillers[variable] = create_silicone_based_infiller(
                    variable=variable,
                    db=self.infilling_configuration[variable].db(),
                    cruncher=self.infilling_configuration[variable].cruncher,
                    lead=self.infilling_configuration[variable].lead,
                )

            infilled_variable = self.infillers[variable](in_emissions)
            infilled_l.append(infilled_variable)

        infilled = pd.concat(infilled_l)

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
                complete_index=self.historical_emissions.index.droplevel(
                    self.unit_level
                ),
            )

        return infilled


def create_silicone_based_infiller(
    variable: str,
    lead: str | tuple[str, ...],
    db: pd.DataFrame,
    cruncher: silicone.database_crunchers.base._DatabaseCruncher,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create a [silicone][]-based infiller (function)

    Parameters
    ----------
    variable
        Variable to infill

    lead
        Lead variable(s) to use for the infilling

    db
        Infilling database

    cruncher
        Cruncher to use for deriving the relationship between `follower` and `lead`

    Returns
    -------
    :
        Infiling function
    """
    try:
        import pyam  # type: ignore # pyam not typed
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "create_silicone_based_infiller", requirement="pyam"
        ) from exc

    if isinstance(lead, tuple):
        lead_use = list(lead)
    else:
        lead_use = [lead]

    infiller_silicone = cruncher(pyam.IamDataFrame(db)).derive_relationship(
        variable_follower=variable,
        variable_leaders=lead_use,
    )

    def infiller(inp: pd.DataFrame) -> pd.DataFrame:
        # TODO: write an alternate package to silicone that doesn't rely on pyam
        # (just use e.g. pandas-indexing instead)
        res_pyamdf = infiller_silicone(pyam.IamDataFrame(inp))
        res = cast(pd.DataFrame, res_pyamdf.timeseries())

        # TODO: The fact that this is needed suggests there's a bug in silicone
        res = res.loc[:, inp.columns]  # type: ignore # pandas-stubs being stupid

        return res

    return infiller
