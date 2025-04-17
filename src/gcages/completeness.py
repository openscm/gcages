"""
General tools for checking completeness
"""

from __future__ import annotations

import pandas as pd


class NotCompleteError(ValueError):
    """
    Raised when a [pd.DataFrame][pandas.DataFrame] is not complete
    """

    def __init__(
        self,
        missing: pd.DataFrame,
        complete_index: pd.MultiIndex,
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        missing
            Index levels that don't have a complete index

        complete_index
            Definition of a complete index
        """
        error_msg = (
            "The DataFrame is not complete. "
            f"The following expected levels are missing:\n{missing}\n"
            f"The complete index expected for each level is:\n"
            f"{complete_index.to_frame(index=False)}"
        )
        super().__init__(error_msg)


def assert_all_groups_are_complete(
    to_check: pd.DataFrame,
    complete_index: pd.MultiIndex,
    group_keys: list[str] | None = None,
    unit_col: str = "unit",
) -> None:
    """
    Assert all groups have 'complete' data

    Here, complete is defined by `complete_index`,
    which specifies the metadata that should be included for each group.

    Parameters
    ----------
    to_check
        Data to check

    complete_index
        Index which defines the meaning of 'complete'

    group_keys
        Keys to use to group `to_check` into groups when checking for completeness.

        If not supplied, we use all the index levels in `to_check`
        except those that appear in `complete_index` and the `unit_col`,
        specifically
        `to_check.index.names.difference([*complete_index.names, unit_col])`.

    unit_col
        Unit column (differences here do not indicate incompleteness)

    Examples
    --------
    >>> to_check = pd.DataFrame(
    ...     [
    ...         [1.0, 2.0],
    ...         [3.0, 2.0],
    ...         [1.0, 2.0],
    ...         [3.0, 2.0],
    ...     ],
    ...     columns=[2015, 2100],
    ...     index=pd.MultiIndex.from_tuples(
    ...         [
    ...             ("sa", "va", "W"),
    ...             ("sa", "vb", "W"),
    ...             ("sb", "va", "W"),
    ...             ("sb", "vb", "W"),
    ...         ],
    ...         names=["scenario", "variable", "unit"],
    ...     ),
    ... )
    >>> to_check  # doctest: +NORMALIZE_WHITESPACE
                            2015  2100
    scenario variable unit
    sa       va       W      1.0   2.0
             vb       W      3.0   2.0
    sb       va       W      1.0   2.0
             vb       W      3.0   2.0

    >>> # A checker, by which `to_check` is complete
    >>> checker_a = pd.MultiIndex.from_tuples(
    ...     [
    ...         ("va",),
    ...         ("vb",),
    ...     ],
    ...     names=["variable"],
    ... )
    >>> assert_all_groups_are_complete(to_check, complete_index=checker_a)
    >>> # No error raised, all happy
    >>>
    >>> # A checker which includes variables that aren't present in `to_check`
    >>> checker_b = pd.MultiIndex.from_tuples(
    ...     [
    ...         ("va",),
    ...         ("vb",),
    ...         ("vc",),
    ...     ],
    ...     names=["variable"],
    ... )
    >>> assert_all_groups_are_complete(to_check, complete_index=checker_b)
    Traceback (most recent call last):
        ...
    gcages.completeness.NotCompleteError: The DataFrame is not complete. The following expected levels are missing:
      variable scenario
    0       vc       sa
    0       vc       sb
    The complete index expected for each level is:
      variable
    0       va
    1       vb
    2       vc
    """  # noqa: E501
    # Probably a smarter way to do this rather than looping, I can't see it now
    if unit_col not in to_check.index.names:
        msg = f"{unit_col=} is not in {to_check.index.names=}"
        raise KeyError(msg)

    if group_keys is None:
        group_keys = to_check.index.names.difference([*complete_index.names, unit_col])  # type: ignore # pandas-stubs confused

    missing_l = []
    # Check against the levels that are in `complete_index`
    # (also ignoring units if they are there)
    idx_to_check_drop_levels = list(
        {*to_check.index.names.difference(complete_index.names), unit_col}  # type: ignore # pandas-stubs confused
    )
    for group_values, gdf in to_check.groupby(group_keys):
        idx_to_check = gdf.index.droplevel(idx_to_check_drop_levels)

        if not isinstance(idx_to_check, pd.MultiIndex):
            idx_to_check = pd.MultiIndex.from_arrays(
                [idx_to_check.values], names=[idx_to_check.name]
            )

        missing_levels = complete_index.difference(  # type: ignore # pandas-stubs out of date
            idx_to_check.reorder_levels(complete_index.names)
        )
        if not missing_levels.empty:
            tmp = missing_levels.to_frame(index=False)
            # Could probably do this better too if we need speed
            for key, value in zip(group_keys, group_values):
                tmp[key] = value

            missing_l.append(tmp)

    if missing_l:
        raise NotCompleteError(pd.concat(missing_l), complete_index=complete_index)
