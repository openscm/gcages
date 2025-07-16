"""
Common tools across different approaches
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from gcages.typing import NUMERIC_DATA, TIME_POINT, TimeseriesDataFrame
from gcages.units_helpers import convert_unit_like

if TYPE_CHECKING:
    import pint


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


def assert_harmonised(  # noqa: PLR0913
    df: TimeseriesDataFrame,
    *,
    history: TimeseriesDataFrame,
    harmonisation_time: TIME_POINT,
    rounding: int = 10,
    df_unit_level: str = "unit",
    history_unit_level: str | None = None,
    ur: pint.UnitRegistry | None = None,
) -> None:
    """
    Assert that the input is harmonised

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

    df_unit_level
        Level in `df`'s index which has unit information

        Only used if unit conversion is required

    history_unit_level
        Level in `history`'s index which has unit information

        If not provided, we assume this is the same as `df_unit_level`

        Only used if unit conversion is required

    ur
        Unit registry to use for determining unit conversions

        Passed to [gcages.units_helpers.convert_unit_like][]

        Only used if unit conversion is required

    Raises
    ------
    NotHarmonisedError
        `df` is not harmonised to `history`
    """
    df_unit_match = convert_unit_like(
        df,
        target=history,
        df_unit_level=df_unit_level,
        target_unit_level=history_unit_level,
        ur=ur,
    )
    df_harm_year_aligned, history_harm_year_aligned = align_history_to_data_at_time(
        df_unit_match, history=history, time=harmonisation_time
    )
    comparison = df_harm_year_aligned.round(rounding).compare(  # type: ignore # pandas-stubs out of date
        history_harm_year_aligned.round(rounding), result_names=("df", "history")
    )
    if not comparison.empty:
        raise NotHarmonisedError(
            comparison=comparison, harmonisation_time=harmonisation_time
        )
