"""
Interpolate to annual function
"""

import pandas as pd


def interpolate_to_annual(
    df: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """
    Add any missing annual columns in [start_year, end_year] and linearly interpolate.

    Trailing NaNs are forward-filled and leading NaNs are left as NaN.

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] containing data with year columns

    start_year
        Initial year for the interpolation

    start_year
        Final year for the interpolation

    Returns
    -------
    :
        Interpolated [pd.DataFrame][pandas.DataFrame]
    """
    if df.columns.empty:
        msg = "`df` has no columns, so there is nothing to interpolate"
        raise ValueError(msg)

    if start_year is None:
        start_year = int(df.columns.min())
    if end_year is None:
        end_year = int(df.columns.max())

    if start_year > end_year:
        msg = f"{start_year=} must be <= {end_year=}"
        raise ValueError(msg)

    columns = sorted(set(df.columns) | set(range(start_year, end_year + 1)))
    if len(columns) == 1:
        # Single time point: nothing to interpolate
        return df.copy()

    # In case of (latest_data_year < end_year) extrapolate a constant value
    return df.reindex(columns=columns).T.interpolate(method="index").T
