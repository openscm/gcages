"""
Aggregation helpers
"""

from __future__ import annotations

import string
from functools import partial
from typing import Callable

import pandas as pd
from pandas_openscm.grouping import groupby_except
from pandas_openscm.indexing import multi_index_lookup, multi_index_match

import gcages.index_manipulation
from gcages.exceptions import MissingOptionalDependencyError
from gcages.index_manipulation import combine_species, set_new_single_value_levels
from gcages.testing import compare_close


def aggregate_df_level(
    indf: pd.DataFrame,
    level: str,
    on_clash: str = "raise",
    level_separator: str = "|",
    min_levels_output: int = 1,
) -> pd.DataFrame:
    # TODO: move this into pandas-openscm
    # will require writing our own extract and format functions
    # will require moving compare_close too
    try:
        from pandas_indexing.core import extractlevel, formatlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "assert_frame_equal", requirement="pandas_indexing"
        ) from exc
    level_groups = {
        n_levels: df
        for n_levels, df in indf.groupby(
            indf.index.get_level_values(level).str.count(rf"\{level_separator}")
        )
    }
    levels_r = range(min_levels_output, max(level_groups) + 1)[::-1]
    # Start by storing the bottom level
    res_d = {levels_r[0]: level_groups[levels_r[0]]}
    for n_levels in levels_r[1:]:
        # Get everything we have already handled
        # at the level below the level of interest
        to_aggregate = res_d[n_levels + 1]

        # Aggregate
        to_aggregate.index = to_aggregate.index.remove_unused_levels()  # type: ignore # pandas-stubs confused

        level_splits = [
            f"{level}_{string.ascii_lowercase[i]}" for i in range(n_levels + 1 + 1)
        ]
        extract_str = level_separator.join(["{" + ls + "}" for ls in level_splits])
        to_aggregate_split = extractlevel(to_aggregate, **{level: extract_str})

        to_aggregate_sum = groupby_except(to_aggregate_split, level_splits[-1]).sum()

        to_aggregate_sum_combined = formatlevel(
            to_aggregate_sum,
            **{
                level: level_separator.join(
                    ["{" + ls + "}" for ls in level_splits[:-1]]
                )
            },
            drop=True,
        )

        keep_at_level = [to_aggregate_sum_combined]
        if n_levels in level_groups:
            # Check if any of the aggregations clash with the input
            indf_at_aggregated_level = level_groups[n_levels]
            clash_locator = multi_index_match(
                indf_at_aggregated_level.index,  # type: ignore # pandas-stubs confused
                to_aggregate_sum_combined.index,
            )
            if not clash_locator.any():
                # No clashing data so simply keep all of `indf_at_aggregated_level`
                keep_at_level.append(indf_at_aggregated_level)

            elif on_clash == "raise":
                clash = indf_at_aggregated_level[clash_locator]
                msg = f"Reaggregated levels are in the input. Clashing levels: {clash}"
                # TODO: switch to custom error
                raise ValueError(msg)

            elif on_clash == "verify":
                indf_compare = indf_at_aggregated_level[clash_locator]
                to_aggregate_sum_combined_compare = multi_index_lookup(
                    to_aggregate_sum_combined,
                    indf_compare.index.remove_unused_levels(),  # type: ignore # pandas-stubs confused
                )
                comparison = compare_close(
                    left=indf_compare,
                    right=to_aggregate_sum_combined_compare,
                    left_name="indf",
                    right_name="aggregated_sum",
                    # **tolerances,
                )
                if not comparison.empty:
                    raise NotImplementedError

            elif on_clash == "overwrite":
                not_clashing = indf_at_aggregated_level[~clash_locator]
                if not_clashing.empty:
                    # (Nothing to keep from input so do nothing here)
                    pass

                else:
                    # Only keep what doesn't clash, effectively overwriting the rest
                    # by using to_aggregate_sum_combined
                    # instead
                    keep_at_level.append(not_clashing)

            else:
                raise NotImplementedError(on_clash)

        res_d[n_levels] = pd.concat(
            [
                df.reorder_levels(to_aggregate_sum_combined.index.names)
                for df in keep_at_level
            ]
        )

    res = pd.concat(
        [
            df.reorder_levels(res_d[min_levels_output].index.names)
            for df in res_d.values()
        ]
    )

    return res


def get_region_sector_sum(
    indf: pd.DataFrame,
    region_level: str = "region",
    world_region: str = "World",
    split_sectors: Callable[[pd.DataFrame], pd.DataFrame] = partial(
        gcages.index_manipulation.split_sectors, bottom_level="sectors"
    ),
    sectors_level: str = "sectors",
) -> pd.DataFrame:
    # Blind sum so double counting is possible if you're not careful
    # Can't avoid double counting as that is defined by data model,
    # not by "|" separation rules
    # Implicitly assumes that `split_sectors` will work
    res = set_new_single_value_levels(
        combine_species(
            groupby_except(split_sectors(indf), [region_level, sectors_level]).sum()
        ),
        {region_level: world_region},
    )

    return res
