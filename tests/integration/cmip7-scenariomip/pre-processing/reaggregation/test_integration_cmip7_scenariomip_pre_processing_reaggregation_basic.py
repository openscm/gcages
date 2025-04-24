"""
Tests of basic reaggregation

'Basic' here means reaggregation assuming that domestic aviation
is reported at the model region level.
There may be other reaggregation methods we need to support,
hence why this is given a specific name.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    # COMPLETE_TIMESERIES_INDEX,
    # NAIVE_SUM_TIMESERIES_INDEX,
    # OPTIONAL_TIMESERIES_INDEX,
    get_required_timeseries_index,
    has_all_required_timeseries,
    # is_internally_consistent,
    # to_complete_timeseries,
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
    # Should pass
    assert False


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
