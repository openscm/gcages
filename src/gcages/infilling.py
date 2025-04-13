"""General infilling tools"""

from __future__ import annotations

import pandas as pd


class NotInfilledError(ValueError):
    """
    Raised when a [pd.DataFrame][pandas.DataFrame] is not infilled
    """

    def __init__(
        self,
        missing: pd.DataFrame,
        full_emissions_index: pd.MultiIndex,
    ) -> None:
        error_msg = (
            "The DataFrame is not fully infilled. "
            f"The following expected levels are missing:\n{missing}\n"
            f"The full index expected for each level is:\n"
            f"{full_emissions_index.to_frame(index=False)}"
        )
        super().__init__(error_msg)


def assert_infilled(
    to_check: pd.DataFrame, full_emissions_index: pd.MultiIndex, unit_col: str = "unit"
) -> None:
    # Probably a smarter way to do this, I can't see it now
    group_keys = to_check.index.names.difference(
        [*full_emissions_index.names, unit_col]
    )
    missing_l = []
    for group_values, gdf in to_check.groupby(group_keys):
        idx_to_check = gdf.index.droplevel([*group_keys, unit_col])
        if isinstance(idx_to_check, pd.Index):
            idx_to_check = pd.MultiIndex.from_arrays(
                [idx_to_check.values], names=[idx_to_check.name]
            )

        missing_levels = full_emissions_index.difference(idx_to_check)
        if not missing_levels.empty:
            tmp = missing_levels.to_frame(index=False)
            for key, value in zip(group_keys, group_values):
                tmp[key] = value

            missing_l.append(tmp)

    if missing_l:
        raise NotInfilledError(
            pd.concat(missing_l), full_emissions_index=full_emissions_index
        )
