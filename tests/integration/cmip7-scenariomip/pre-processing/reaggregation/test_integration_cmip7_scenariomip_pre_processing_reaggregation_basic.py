"""
Tests of basic reaggregation

'Basic' here means reaggregation assuming that domestic aviation
is reported at the model region level.
There may be other reaggregation methods we need to support,
hence why this is given a specific name.
"""

from __future__ import annotations

import itertools
import string

import numpy as np
import pandas as pd
import pytest
from pandas_openscm.grouping import groupby_except
from pandas_openscm.indexing import multi_index_lookup

from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    get_internal_consistency_checking_index,
    # to_complete_timeseries,
    # COMPLETE_TIMESERIES_INDEX,
    # NAIVE_SUM_TIMESERIES_INDEX,
    # OPTIONAL_TIMESERIES_INDEX,
    get_required_timeseries_index,
    has_all_required_timeseries,
    is_internally_consistent,
)
from gcages.completeness import NotCompleteError
from gcages.index_manipulation import set_new_single_value_levels
from gcages.typing import NP_ARRAY_OF_FLOAT_OR_INT

RNG = np.random.default_rng()


def get_df(
    base_index: pd.MultiIndex,
    model: str,
    timepoints: NP_ARRAY_OF_FLOAT_OR_INT = np.arange(2010, 2100 + 1, 10.0),
    scenario: str = "scenario_a",
) -> pd.DataFrame:
    res = set_new_single_value_levels(
        pd.DataFrame(
            RNG.random((base_index.shape[0], timepoints.size)),
            columns=timepoints,
            index=base_index,
        ),
        {"model": model, "scenario": scenario, "unit": "Mt"},
    )

    return res


def aggregate_df(indf: pd.DataFrame) -> pd.DataFrame:
    # TODO: move this into pandas-openscm
    pix = pytest.importorskip("pandas_indexing")

    level_separator = "|"
    level_to_aggregate = "variable"
    on_clash = "raise"
    min_levels_output = 1

    level_groups = {
        n_levels: df
        for n_levels, df in indf.groupby(
            indf.index.get_level_values(level_to_aggregate).str.count(
                rf"\{level_separator}"
            )
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
        to_aggregate.index = to_aggregate.index.remove_unused_levels()

        level_splits = [
            f"{level_to_aggregate}_{string.ascii_lowercase[i]}"
            for i in range(n_levels + 1 + 1)
        ]
        extract_str = level_separator.join(["{" + ls + "}" for ls in level_splits])
        to_aggregate_split = to_aggregate.pix.extract(
            **{level_to_aggregate: extract_str}
        )

        to_aggregate_sum = groupby_except(to_aggregate_split, level_splits[-1]).sum()

        to_aggregate_sum_combined = to_aggregate_sum.pix.format(
            **{
                level_to_aggregate: level_separator.join(
                    ["{" + ls + "}" for ls in level_splits[:-1]]
                )
            },
            drop=True,
        )

        keep_at_level = [to_aggregate_sum_combined]
        if n_levels in level_groups:
            # Check if any of the aggregations clash with the input
            indf_at_aggregated_level = level_groups[n_levels]
            clash = multi_index_lookup(
                indf_at_aggregated_level, to_aggregate_sum_combined.index
            )
            if clash.empty:
                # No clashing data so simply keep all of `indf_at_aggregated_level`
                keep_at_level.append(indf_at_aggregated_level)

            elif on_clash == "raise":
                msg = f"Reaggregated levels are in the input. Clashing levels: {clash}"
                # TODO: switch to custom error
                raise ValueError(msg)

            # elif on_clash == "verify":
            # make sure that the aggregation is the same as what is already there
            # in level_groups[n_levels - 1],
            # raise if there is a difference

            # elif on_clash == "overwrite":
            # edit level_groups[n_levels - 1]
            # to overwrite the clashing data

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


def get_aggregate_df(indf: pd.DataFrame, world_region: str) -> pd.DataFrame:
    # The way the data is specified, a blind sum over regions is fine.
    df_regional = indf.loc[indf.index.get_level_values("region") != world_region]
    df_region_sum = groupby_except(indf, "region").sum()

    df_region_sum_aggregated = set_new_single_value_levels(
        aggregate_df(df_region_sum), {"region": world_region}
    )
    res = pd.concat(
        [df_regional, df_region_sum_aggregated.reorder_levels(df_regional.index.names)]
    )

    return res


def test_has_all_required_timeseries_full_dataset():
    model = "model_a"
    model_regions = [f"{model}|{r}" for r in ["Pacific OECD", "China"]]

    df = get_df(
        base_index=get_required_timeseries_index(model_regions),
        model=model,
    )

    assert has_all_required_timeseries(df, model_regions=model_regions) is None


def test_has_all_required_timeseries_extra_timeseries():
    model = "model_a"
    model_regions = [f"{model}|{r}" for r in ["Pacific OECD", "China"]]

    base_index = get_required_timeseries_index(model_regions).append(
        pd.MultiIndex.from_tuples(
            [
                ("junk", "World"),
                ("junk", f"{model}|China"),
                ("more junk", "model|not a region"),
                ("Emissions|CH4|Transport|Something", "World"),
            ],
            names=["variable", "region"],
        )
    )

    df = get_df(base_index=base_index, model=model)

    assert has_all_required_timeseries(df, model_regions=model_regions) is None


@pytest.mark.parametrize(
    "complete_index, to_remove, model, model_regions",
    (
        pytest.param(complete_index, row, model, model_regions)
        for model in ["model_a"]
        for model_regions in [[f"{model}|{r}" for r in ["Pacific OECD", "China"]]]
        for complete_index in [get_required_timeseries_index(model_regions)]
        for row in [
            *complete_index,
            # Just get a selection of combos of dropped elements
            *[list(v) for v in list(itertools.combinations(complete_index, 2))[:20]],
        ]
    ),
)
def test_has_all_required_timeseries_missing_timeseries(
    complete_index, to_remove, model, model_regions
):
    df = get_df(base_index=complete_index.drop(to_remove), model=model)

    with pytest.raises(NotCompleteError):
        has_all_required_timeseries(df, model_regions=model_regions)


def test_is_internally_consistent_correct_dataset():
    model = "model_a"
    model_regions = [f"{model}|{r}" for r in ["Pacific OECD", "China"]]
    world_region = "World"

    df = get_df(
        base_index=get_internal_consistency_checking_index(
            model_regions, world_region=world_region
        ),
        model=model,
    )

    df_to_check = get_aggregate_df(df, world_region=world_region)

    assert is_internally_consistent(df_to_check, model_regions=model_regions) is None


def test_is_internally_consistent_correct_dataset_missing_timeseries():
    # Should pass
    assert False
    is_internally_consistent()


def test_is_internally_consistent_correct_dataset_extra_timeseries():
    # Should pass
    assert False


def test_is_internally_consistent_incorrect_dataset():
    # Should pass
    assert False


def test_is_internally_consistent_incorrect_dataset_missing_timeseries():
    # Should pass
    assert False


def test_is_internally_consistent_incorrect_dataset_extra_timeseries():
    # Should pass
    assert False


def test_to_complete_from_full_dataset():
    # start with the full index
    # call to complete
    # should get back the full index
    assert False


def test_to_complete_extra_timeseries():
    # Should get just the complete bits back
    assert False
    to_complete_timeseries()


def test_to_complete_missing_timeseries():
    # Should get the optional timeseries as zeros
    assert False


def test_to_complete_extra_and_missing_optional_timeseries():
    # Should get just the complete bits back
    # Optional timeseries filled with zeros
    assert False
