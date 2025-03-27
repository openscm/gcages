"""
Useful assertions
"""

from __future__ import annotations

import pandas as pd


def assert_only_working_on_variable_unit_variations(indf: pd.DataFrame) -> None:
    """
    Assert that we're only working on variations in variable and unit

    In other words, we don't have variations in scenarios, models etc.

    Parameters
    ----------
    indf
        Data to verify

    Raises
    ------
    AssertionError
        There are variations in columns other than variable and unit
    """
    variations_in_other_cols = indf.index.droplevel(["variable", "unit"]).unique()
    if len(variations_in_other_cols) > 1:
        raise AssertionError(f"{variations_in_other_cols=}")
