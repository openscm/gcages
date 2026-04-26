"""
Functionality that should be moved into [pandas_openscm][]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def interpolate_to_annual_timesteps(indf: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate to annual timesteps

    Parameters
    ----------
    indf
        Input data

    Returns
    -------
    :
        `indf`, interpolated to annual timesteps
    """
    # TODO: add checks
    yearly_timesteps = np.arange(indf.columns.min(), indf.columns.max() + 1)
    res = indf.reindex(columns=yearly_timesteps).T.interpolate(method="index").T

    return res
