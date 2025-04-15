"""
General post-processing tools
"""

from __future__ import annotations

import pandas as pd
from attrs import define


@define
class PostProcessingResult:
    """
    Results of post-processing

    The data is separated into tables that have the same index levels.
    This simplifies further processing
    (at the expense of making it more complex to get data into the object).
    """

    timeseries_run_id: pd.DataFrame
    """Timeseries that includes a run_id index level"""

    timeseries_quantile: pd.DataFrame
    """Timeseries that includes a quantile index level"""

    timeseries_exceedance_probabilities: pd.DataFrame
    """
    Timeseries of exceedance probabilities

    These are reported separately because they contain
    extra index levels to handle the threshold information.
    """

    metadata_run_id: pd.Series[float]
    """Metadata that includes a run_id index level"""

    metadata_quantile: pd.Series[float]
    """Metadata that includes a quantile index level"""

    metadata_exceedance_probabilities: pd.Series[float]
    """
    Metadata of exceedance probabilities

    These are reported separately because they contain
    extra index levels to handle the threshold information.
    """

    metadata_categories: pd.Series[str]
    """
    Metadata of categories
    """
