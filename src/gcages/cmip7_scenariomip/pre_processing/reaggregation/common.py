"""
Common components used across different re-aggregation strategies
"""

from __future__ import annotations

import pandas as pd
from attrs import define


@define
class ToCompleteResult:
    """
    Result of calling `to_complete` on a reaggregator
    """

    complete: pd.DataFrame
    """Complete [pd.DataFrame][pandas.DataFrame]"""

    assumed_zero: pd.DataFrame | None
    """
    The timeseries that were assumed to be zero to make `self.complete`

    If `None`, no timeseries were assumed to be zero.
    """
