"""
Internal consistency checking helpers
"""

# TODO: move this to pandas-openscm
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


class InternalConsistencyError(ValueError):
    """
    Raised when there is an internal consistency issue in the data

    Specifically, the sum of components doesn't match some total
    """

    def __init__(
        self,
        differences: pd.DataFrame,
        data_that_was_summed: pd.DataFrame,
        tolerances: Mapping[str, float],
    ) -> None:
        differences_variables = differences.index.get_level_values("variable").unique()
        data_that_was_summed_relevant_for_differences = data_that_was_summed[
            data_that_was_summed.index.get_level_values("variable").map(
                lambda x: any(v in x for v in differences_variables)
            )
        ].index.to_frame(index=False)

        error_msg = (
            "Summing the components does not equal the total "
            f"when using the following tolerances: {tolerances}. "
            f"Differences:\n{differences}\n"
            "This is the data we used in the sum:\n"
            f"{data_that_was_summed_relevant_for_differences}"
        )

        super().__init__(error_msg)
