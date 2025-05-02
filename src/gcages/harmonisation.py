"""General harmonisation tools"""

from __future__ import annotations

import importlib
import multiprocessing
from typing import Any

import attr
import pandas as pd
from attrs import define, field
from pandas_openscm.parallelisation import ParallelOpConfig, apply_op_parallel_progress

from gcages.aneris_helpers import harmonise_all
from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_data_for_times,
    assert_has_index_levels,
    assert_index_is_multiindex,
    assert_metadata_values_all_allowed,
    assert_only_working_on_variable_unit_variations,
)
from gcages.exceptions import MissingOptionalDependencyError
from gcages.typing import NUMERIC_DATA, TIME_POINT, TimeseriesDataFrame


class NotHarmonisedError(ValueError):
    """
    Raised when a [pd.DataFrame][pandas.DataFrame] is not harmonised
    """

    def __init__(
        self,
        comparison: pd.DataFrame,
        harmonisation_time: TIME_POINT,
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        comparison
            Results of comparing the data and history

        harmonisation_time
            Expected harmonisation time
        """
        error_msg = (
            f"The DataFrame is not harmonised in {harmonisation_time}. "
            f"comparison=\n{comparison}"
        )
        super().__init__(error_msg)


def align_history_to_data_at_time(
    df: TimeseriesDataFrame, *, history: TimeseriesDataFrame, time: Any
) -> tuple[pd.Series[NUMERIC_DATA], pd.Series[NUMERIC_DATA]]:  # type: ignore # pandas-stubs not up to date
    """
    Align history to a given set of data for a given column

    Parameters
    ----------
    df
        Data to which to align history

    history
        History data to align

    time
        Time (i.e. column) for which to align the data

    Returns
    -------
    :
        History, aligned with `df` for the given column

    Raises
    ------
    AssertionError
        `df` and `history` could not be aligned for some reason
    """
    df_year_aligned, history_year_aligned = df[time].align(history[time], join="left")

    # Implicitly assuming that people have already checked
    # that they have history values for all timeseries in `df`,
    # so any null is an obvious issue.
    if history_year_aligned.isnull().any():
        msg_l = ["history did not align properly with df"]

        if df.index.names == history.index.names:
            msg_l.append(
                "history and df have the same index levels "
                f"({list(history.index.names)}). "
                "You probably need to drop some of history's index levels "
                "so alignment can happen along the levels of interest "
                "(usually dropping everything except variable and unit (or similar)). "
            )

        # # Might be useful, pandas might handle it
        # names_only_in_hist = history.index.names.difference(df.index.names)

        for unit_col_guess in ["unit", "units"]:
            if (
                unit_col_guess in df.index.names
                and unit_col_guess in history.index.names
            ):
                df_units_guess = df.index.get_level_values(unit_col_guess)
                history_units_guess = history.index.get_level_values(unit_col_guess)

                differing_units = (
                    df_units_guess.difference(history_units_guess).unique().tolist()
                )
                msg_l.append(
                    "The following units only appear in `df`, "
                    f"which might be why the data isn't aligned: {differing_units}. "
                    f"{df_units_guess=} {history_units_guess=}"
                )

        msg = ". ".join(msg_l)
        raise AssertionError(msg)

    return df_year_aligned, history_year_aligned


def assert_harmonised(
    df: TimeseriesDataFrame,
    *,
    history: TimeseriesDataFrame,
    harmonisation_time: TIME_POINT,
    rounding: int = 10,
) -> None:
    """
    Assert that a given [TimeseriesDataFrame][(p).typing] is harmonised

    Note: currently, this does not support unit conversion
    (i.e. units have to match exactly, equivalent units e.g. "Mt CO2" and "MtCO2"
    will result in a `NotHarmonisedError`).

    Parameters
    ----------
    df
        Data to check

    history
        History to which `df` should be harmonised

    harmonisation_time
        Time at which `df` should be harmonised to `history`

    rounding
        Rounding to apply to the data before comparing

    Raises
    ------
    NotHarmonisedError
        `df` is not harmonised to `history`
    """
    df_harm_year_aligned, history_harm_year_aligned = align_history_to_data_at_time(
        df, history=history, time=harmonisation_time
    )
    comparison = df_harm_year_aligned.round(rounding).compare(  # type: ignore # pandas-stubs out of date
        history_harm_year_aligned.round(rounding), result_names=("df", "history")
    )
    if not comparison.empty:
        raise NotHarmonisedError(
            comparison=comparison, harmonisation_time=harmonisation_time
        )


def harmonise_scenario(
    indf: pd.DataFrame,
    history: pd.DataFrame,
    year: int,
    overrides: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Harmonise a single scenario

    Parameters
    ----------
    indf
        Scenario to harmonise

    history
        History to harmonise to

    year
        Year to use for harmonisation

    overrides
        Overrides to pass to aneris

    Returns
    -------
    :
        Harmonised scenario
    """
    if importlib.util.find_spec("scipy") is None:
        raise MissingOptionalDependencyError("harmonise_scenario", requirement="scipy")

    assert_only_working_on_variable_unit_variations(indf)

    harmonised = harmonise_all(
        indf,
        history=history,
        year=year,
        overrides=overrides,
    )

    return harmonised


@define
class AnerisHarmoniser:
    """
    Harmoniser that uses [aneris][]
    """

    historical_emissions: pd.DataFrame = field()
    """
    Historical emissions to use for harmonisation
    """

    harmonisation_year: int
    """
    Year in which to harmonise
    """

    aneris_overrides: pd.DataFrame | None = field(default=None)
    """
    Overrides to supply to `aneris.convenience.harmonise_all`

    For source code and docs,
    see e.g. [https://github.com/iiasa/aneris/blob/v0.4.2/src/aneris/convenience.py]().
    """

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    variable_level: str = "variable"
    """
    Level in data indexes that represents the variable of the timeseries
    """

    region_level: str = "region"
    """
    Level in data indexes that represents the region of the timeseries
    """

    unit_level: str = "unit"
    """
    Level in data indexes that represents the unit of the timeseries
    """

    scenario_group_levels: list[str] = field(factory=lambda: ["model", "scenario"])
    """
    Levels in data indexes to use to group data into scenarios

    Here, 'scenarios' means groups of timeseries
    that will be run through a climate model.
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

    @aneris_overrides.validator
    def validate_aneris_overrides(
        self, attribute: attr.Attribute[Any], value: pd.DataFrame | None
    ) -> None:
        """
        Validate the aneris overrides value

        If `self.run_checks` is `False`, then this is a no-op
        """
        if value is None:
            return

        if not self.run_checks:
            return

        value_check = pd.DataFrame(
            value["method"].values,
            columns=["method"],
            index=pd.MultiIndex.from_frame(
                value[value.columns.difference(["method"]).tolist()]
            ),
        )
        for index_level in self.historical_emissions.index.names:
            assert_metadata_values_all_allowed(
                value_check,
                metadata_key=index_level,
                allowed_values=self.historical_emissions.index.get_level_values(
                    index_level
                ).unique(),
            )

    @historical_emissions.validator
    def validate_historical_emissions(
        self, attribute: attr.Attribute[Any], value: pd.DataFrame
    ) -> None:
        """
        Validate the historical emissions value

        If `self.run_checks` is `False`, then this is a no-op
        """
        if not self.run_checks:
            return

        assert_index_is_multiindex(value)
        assert_data_is_all_numeric(value)
        assert_has_index_levels(
            value, [self.variable_level, self.region_level, self.unit_level]
        )
        assert_has_data_for_times(
            value, times=[self.harmonisation_year], allow_nan=False
        )

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonise

        Parameters
        ----------
        in_emissions
            Emissions to harmonise

        Returns
        -------
        :
            Harmonised emissions
        """
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
            assert_has_data_for_times(
                in_emissions, times=[self.harmonisation_year], allow_nan=False
            )

            assert_metadata_values_all_allowed(
                in_emissions,
                metadata_key=self.variable_level,
                allowed_values=self.historical_emissions.index.get_level_values(
                    self.variable_level
                ).unique(),
            )

        harmonised_df = pd.concat(
            apply_op_parallel_progress(
                func_to_call=harmonise_scenario,
                iterable_input=(
                    gdf for _, gdf in in_emissions.groupby(self.scenario_group_levels)
                ),
                parallel_op_config=ParallelOpConfig.from_user_facing(
                    progress=self.progress,
                    max_workers=self.n_processes,
                    progress_results_kwargs=dict(desc="Scenarios to harmonise"),
                ),
                history=self.historical_emissions,
                year=self.harmonisation_year,
                overrides=self.aneris_overrides,
            )
        )

        if self.run_checks:
            assert_harmonised(
                harmonised_df,
                history=self.historical_emissions,
                harmonisation_time=self.harmonisation_year,
            )

            pd.testing.assert_index_equal(
                harmonised_df.index,
                in_emissions.index,
                check_order=False,  # type: ignore # pandas-stubs out of date
            )
            if harmonised_df.columns.dtype != in_emissions.columns.dtype:
                msg = (
                    "Column type has changed: "
                    f"{harmonised_df.columns.dtype=} {in_emissions.columns.dtype=}"
                )
                raise AssertionError(msg)

        return harmonised_df
