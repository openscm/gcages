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
from collections.abc import Iterable
from contextlib import nullcontext as does_not_raise
from functools import partial

import numpy as np
import pandas as pd
import pytest
from attrs import define
from pandas_openscm.grouping import groupby_except
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.indexing import multi_index_lookup, multi_index_match

from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    get_internal_consistency_checking_index,
    # to_complete_timeseries,
    get_required_timeseries_index,
    has_all_required_timeseries,
    is_internally_consistent,
)
from gcages.completeness import NotCompleteError
from gcages.index_manipulation import set_new_single_value_levels
from gcages.internal_consistency import InternalConsistencyError
from gcages.typing import NP_ARRAY_OF_FLOAT_OR_INT

RNG = np.random.default_rng()

# TODO: function for generating aggregated case using caching for speed

# TODO: change tests of internally consistent to use parameterisation
#       1. internally consistent as far as we are concerned
#          but not internally consistent from hierarchy point of view: does_not_raise
#         - missing optional timeseries
#         - with extra timeseries
#       1. internally consistent from hierarchy point of view
#          but not as far as we are concerned: raises InternalConsistencyError
#         - missing optional timeseries
#         - with extra timeseries
# TODO: use pint for tolerances

# TODO: change tests of to complete to use parameterisation

# Tests of complete to gridding sectors
# Test of gridding sectors to global workflow go elsewhere

COMPLETE_GRIDDING_SPECIES: tuple[str, ...] = (
    "CO2",
    "CH4",
    "N2O",
    "BC",
    "CO",
    "NH3",
    "OC",
    "NOx",
    "Sulfur",
    "VOC",
)


@define
class GriddingSectorComponentsReporting:
    gridding_sector: str

    spatial_resolution: str

    input_sectors: tuple[str, ...]

    input_sectors_optional: tuple[str, ...]

    input_species_optional: tuple[str, ...]

    def to_complete_variables(self, all_species: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            f"Emissions|{species}|{sector}"
            for species in all_species
            for sector in self.input_sectors
        )

    def to_required_variables(self, all_species: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            f"Emissions|{species}|{sector}"
            for species in all_species
            for sector in self.input_sectors
            if not (
                # note the OR logic here
                sector in self.input_sectors_optional
                or species in self.input_species_optional
            )
        )


GRIDDING_SECTORS = (
    GriddingSectorComponentsReporting(
        gridding_sector="Agriculture",
        spatial_resolution="model region",
        input_sectors=("AFOLU|Agriculture",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Agricultural Waste Burning",
        spatial_resolution="model region",
        input_sectors=(
            "AFOLU|Agricultural Waste Burning",
            "AFOLU|Land|Harvested Wood Products",
            "AFOLU|Land|Land Use and Land-Use Change",
            "AFOLU|Land|Other",
            "AFOLU|Land|Wetlands",
        ),
        input_sectors_optional=(
            "AFOLU|Land|Harvested Wood Products",
            "AFOLU|Land|Land Use and Land-Use Change",
            "AFOLU|Land|Other",
            "AFOLU|Land|Wetlands",
        ),
        input_species_optional=(
            "BC",
            "CO",
            "OC",
            "Sulfur",
        ),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Aircraft",
        spatial_resolution="world",
        input_sectors=(
            "Energy|Demand|Bunkers|International Aviation",
            # Domestic aviation is included too.
            # However, it has to be reported at the regional level
            # so we can subtract it from Transport
            # (hence it doesn't appear here, see below)
        ),
        input_sectors_optional=(),
        input_species_optional=("CH4",),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Domestic aviation headache",
        spatial_resolution="model region",
        input_sectors=("Energy|Demand|Transportation|Domestic Aviation",),
        input_sectors_optional=(),
        input_species_optional=("CH4",),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Transportation Sector",
        spatial_resolution="model region",
        input_sectors=("Energy|Demand|Transportation",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Energy Sector",
        spatial_resolution="model region",
        input_sectors=("Energy|Supply",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Forest Burning",
        spatial_resolution="model region",
        input_sectors=("AFOLU|Land|Fires|Forest Burning",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Grassland Burning",
        spatial_resolution="model region",
        input_sectors=("AFOLU|Land|Fires|Grassland Burning",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Industrial Sector",
        spatial_resolution="model region",
        input_sectors=(
            "Energy|Demand|Industry",
            "Energy|Demand|Other Sector",
            "Industrial Processes",
            "Other",
            "Other Capture and Removal",
        ),
        input_sectors_optional=(
            "Energy|Demand|Other Sector",
            "Other",
            "Other Capture and Removal",
        ),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="International Shipping",
        spatial_resolution="world",
        input_sectors=("Energy|Demand|Bunkers|International Shipping",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Peat Burning",
        spatial_resolution="model region",
        input_sectors=("AFOLU|Land|Fires|Peat Burning",),
        input_sectors_optional=("AFOLU|Land|Fires|Peat Burning",),
        input_species_optional=COMPLETE_GRIDDING_SPECIES,
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Residential Commercial Other",
        spatial_resolution="model region",
        input_sectors=("Energy|Demand|Residential and Commercial and AFOFI",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Solvents Production and Application",
        spatial_resolution="model region",
        input_sectors=("Product Use",),
        input_sectors_optional=(),
        input_species_optional=("BC", "CH4", "CO", "NOx", "OC", "Sulfur"),
    ),
    GriddingSectorComponentsReporting(
        gridding_sector="Waste",
        spatial_resolution="model region",
        input_sectors=("Waste",),
        input_sectors_optional=(),
        input_species_optional=(),
    ),
)


def to_index(  # noqa: PLR0913
    gss: Iterable[GriddingSectorComponentsReporting],
    complete: bool,
    all_species: tuple[str, ...],
    world_region: str | None = None,
    model_regions: tuple[str, ...] | None = None,
    variable_level: str = "variable",
    region_level: str = "region",
) -> pd.MultiIndex:
    res = None
    for gs in gss:
        if complete:
            variables = gs.to_complete_variables(all_species=all_species)
        else:
            variables = gs.to_required_variables(all_species=all_species)

        if gs.spatial_resolution == "model region":
            regions = model_regions
        elif gs.spatial_resolution == "world":
            regions = [world_region]
        else:
            raise NotImplementedError(gs.spatial_resolution)

        gs_index = pd.MultiIndex.from_product(
            [variables, regions], names=[variable_level, region_level]
        )
        if res is None:
            res = gs_index
        else:
            res = res.append(gs_index)

    return res


# For most of the tests, use the same world and model regions.
# Can obviously parameterise differently in individual tests.
WORLD_REGION = "World"
MODEL = "model_a"
MODEL_REGIONS = [f"{MODEL}|{r}" for r in ["Pacific OECD", "China"]]

COMPLETE_INDEX = to_index(
    GRIDDING_SECTORS,
    complete=True,
    all_species=COMPLETE_GRIDDING_SPECIES,
    world_region=WORLD_REGION,
    model_regions=MODEL_REGIONS,
)

REQUIRED_INDEX = to_index(
    GRIDDING_SECTORS,
    complete=False,
    all_species=COMPLETE_GRIDDING_SPECIES,
    world_region=WORLD_REGION,
    model_regions=MODEL_REGIONS,
)

OPTIONAL_INDEX = COMPLETE_INDEX.difference(REQUIRED_INDEX)


def guess_unit(v: str) -> str:
    species = v.split("|")[1]
    unit_map = {
        "BC": "Mt BC/yr",
        "CH4": "Mt CH4/yr",
        "CO": "Mt CO/yr",
        "CO2": "Mt CO2/yr",
        "N2O": "kt N2O/yr",
        "NH3": "Mt NH3/yr",
        "NOx": "Mt NO2/yr",
        "OC": "Mt OC/yr",
        "Sulfur": "Mt SO2/yr",
        "VOC": "Mt VOC/yr",
    }
    return unit_map[species]


def get_df(  # noqa: PLR0913
    base_index: pd.MultiIndex,
    model: str,
    timepoints: NP_ARRAY_OF_FLOAT_OR_INT = np.arange(2010, 2100 + 1, 10.0),
    scenario: str = "scenario_a",
    model_level: str = "model",
    scenario_level: str = "scenario",
    unit_level: str = "unit",
    variable_level: str = "variable",
) -> pd.DataFrame:
    res = set_new_single_value_levels(
        pd.DataFrame(
            RNG.random((base_index.shape[0], timepoints.size)),
            columns=timepoints,
            index=base_index,
        ),
        {model_level: model, scenario_level: scenario},
    )
    # TODO: split this into pandas-openscm
    # (smarter version of `update_levels`
    # i.e. `add_levels_based_on_existing`)

    # Step 1: copy existing variable into new level in index
    variable_level_idx = res.index.names.index(variable_level)
    res.index = pd.MultiIndex(
        levels=[*res.index.levels, res.index.levels[variable_level_idx]],
        codes=[*res.index.codes, res.index.codes[variable_level_idx]],
        names=[*res.index.names, unit_level],
    )
    # Step 2: Apply the map
    res = update_index_levels_func(
        res,
        {unit_level: guess_unit},
        copy=False,
    )

    return res


COMPLETE_DF = get_df(
    base_index=COMPLETE_INDEX,
    model=MODEL,
)

tuples_to_multi_index_vr = partial(
    pd.MultiIndex.from_tuples, names=["variable", "region"]
)


@pytest.mark.parametrize(
    "to_remove, to_add, exp",
    (
        pytest.param(None, None, does_not_raise(), id="complete-df"),
        pytest.param(
            OPTIONAL_INDEX,
            None,
            does_not_raise(),
            id="missing-all-optional",
        ),
        *(
            pytest.param(
                row,
                None,
                does_not_raise(),
                id=f"missing-optional_{row[0][0]}_{row[0][1]}",
            )
            for row in [OPTIONAL_INDEX[[i]] for i in range(OPTIONAL_INDEX.shape[0])]
        ),
        *(
            pytest.param(
                rows,
                None,
                does_not_raise(),
                id="missing-multiple-optional-rows",
            )
            for rows in [OPTIONAL_INDEX[:4:2], OPTIONAL_INDEX[-9::3]]
        ),
        *(
            pytest.param(
                row,
                None,
                pytest.raises(NotCompleteError, match=rf"{row[0][0]}\s*{row[0][1]}"),
                id=f"missing-required_{row[0][0]}_{row[0][1]}",
            )
            for row in [REQUIRED_INDEX[[i]] for i in range(REQUIRED_INDEX.shape[0])]
        ),
        *(
            pytest.param(
                rows,
                None,
                pytest.raises(NotCompleteError),
                id="missing-multiple-required-rows",
            )
            for rows in [REQUIRED_INDEX[:4:2], REQUIRED_INDEX[-9::3]]
        ),
        *(
            pytest.param(
                to_remove,
                to_add,
                does_not_raise(),
                id=f"{to_remove_id}-with-{to_add_id}",
            )
            for (to_remove, to_remove_id), (to_add, to_add_id) in itertools.product(
                [(None, "complete"), (OPTIONAL_INDEX, "required")],
                [
                    (
                        tuples_to_multi_index_vr(
                            [
                                (
                                    "Emissions|CO2|Energy|Demand|Transportation|Rail",
                                    "World",
                                )
                            ]
                        ),
                        "single-extra",
                    ),
                    (
                        tuples_to_multi_index_vr(
                            [
                                *(
                                    ("Emissions|CH4", f"{MODEL}|{r}")
                                    for r in MODEL_REGIONS
                                ),
                            ]
                        ),
                        "multiple-extras",
                    ),
                    (
                        tuples_to_multi_index_vr(
                            [
                                ("Emissions|CH4", "World"),
                                *(
                                    ("Emissions|NOx|Energy", f"{MODEL}|{r}")
                                    for r in MODEL_REGIONS
                                ),
                            ]
                        ),
                        "complex-extras",
                    ),
                ],
            )
        ),
    ),
)
def test_has_all_required_timeseries(to_remove, to_add, exp):
    """
    Tests of `has_all_required_timeseries`

    The combinatorics of this are difficult
    (you can't test all possible combinations of missing things).
    Our strategy is to make sure that:

    1. a complete set of data passes
    1. only the required passes
    1. missing a single optional row passes
    1. missing more than one single optional row passes
       (but not all combinations)
    1. missing any required row raises
    1. missing multiple required rows raises
       (but not all combinations)
    1. extra rows don't cause an issue (whether single or multiple)
    """
    base_df = COMPLETE_DF
    model_regions = MODEL_REGIONS

    to_check = base_df.copy()
    if to_remove is not None:
        to_remove_locator = multi_index_match(to_check.index, to_remove)
        assert to_remove_locator.sum() > 0, "Test won't do anything"
        to_check = to_check.loc[~to_remove_locator]

    if to_add is not None:
        already_included_locator = multi_index_match(to_check.index, to_add)
        assert already_included_locator.sum() == 0, "Test won't do anything"
        to_add_df = get_df(to_add, model=MODEL)

        to_check = pd.concat([to_check, to_add_df.reorder_levels(to_check.index.names)])

    with exp:
        has_all_required_timeseries(to_check, model_regions=model_regions)


def aggregate_df(indf: pd.DataFrame) -> pd.DataFrame:
    # TODO: move this into pandas-openscm
    pytest.importorskip("pandas_indexing")

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
        pytest.param(complete_index, to_remove, model, model_regions)
        for model in ["model_a"]
        for model_regions in [[f"{model}|{r}" for r in ["Pacific OECD", "China"]]]
        for complete_index in [get_required_timeseries_index(model_regions)]
        for to_remove in [
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


@pytest.mark.parametrize(
    [
        "internal_consistency_checking_index",
        "to_remove",
        "model",
        "model_regions",
        "world_region",
    ],
    (
        pytest.param(
            internal_consistency_checking_index,
            to_remove,
            model,
            model_regions,
            world_region,
        )
        for model in ["model_a"]
        for model_regions in [[f"{model}|{r}" for r in ["Pacific OECD", "China"]]]
        for world_region in ["World"]
        for internal_consistency_checking_index in [
            get_internal_consistency_checking_index(
                model_regions, world_region=world_region
            )
        ]
        for optional_index in [
            internal_consistency_checking_index.difference(
                get_required_timeseries_index(
                    model_regions=model_regions, world_region=world_region
                )
            )
        ]
        for to_remove in [
            *optional_index,
            # Get a selection of combos of dropped elements
            *[list(v) for v in list(itertools.combinations(optional_index, 2))[:5]],
        ]
    ),
)
def test_is_internally_consistent_correct_dataset_missing_timeseries(
    internal_consistency_checking_index, to_remove, model, model_regions, world_region
):
    """
    Check that `is_internally_consistent` does not raise for missing optional timeseries

    The data still has to be internally consistent, of course.

    We don't need to test what happens if required timeseries are missing
    because that is caught by `has_all_required_timeseries`.
    """
    base_index = internal_consistency_checking_index.drop(to_remove)

    df = get_df(base_index=base_index, model=model)

    df_to_check = get_aggregate_df(df, world_region=world_region)

    assert is_internally_consistent(df_to_check, model_regions=model_regions) is None


@pytest.mark.parametrize(
    "model, extra_variable_regions",
    (
        (
            "model_a",
            (
                # Domestic aviation can be added late
                # because it's not used in checking the internal consistency
                # (it's effectively shadowed by **Transportation)
                [
                    "Emissions|CO2|Energy|Demand|Transportation|Domestic Aviation",
                    "model_a|China",
                ],
                [
                    "Emissions|CO2|Energy|Demand|Transportation|Domestic Aviation",
                    "model_a|Pacific OECD",
                ],
            ),
        ),
        (
            "model_a",
            (
                ["Emissions|CH4|Energy|Demand|Transportation|Rail", "model_a|China"],
                [
                    "Emissions|CH4|Energy|Demand|Transportation|Rail",
                    "model_a|Pacific OECD",
                ],
            ),
        ),
        (
            "model_a",
            (["Emissions|CO2|Unused", "World"],),
        ),
        (
            "model_a",
            (
                ["Emissions|N2O|Energy|Demand|Transportation|Rail", "model_a|China"],
                [
                    "Emissions|N2O|Energy|Demand|Transportation|Rail",
                    "model_a|Pacific OECD",
                ],
                ["Emissions|N2O|Unused", "World"],
            ),
        ),
    ),
)
def test_is_internally_consistent_correct_dataset_extra_timeseries(
    model, extra_variable_regions
):
    """
    Check that `is_internally_consistent` does not raise
    if we add extra timeseries that break the hierarchy
    but only with variables that aren't actually used for the gridding.

    In other words, check that we check what we're interested in,
    not the internal consistency of the entire dataset which is undefined
    because the internal consistency rules are undefined.
    """
    scenario = "scenario_a"
    model_regions = [f"{model}|{r}" for r in ["Pacific OECD", "China"]]
    world_region = "World"

    df = get_df(
        base_index=get_internal_consistency_checking_index(
            model_regions, world_region=world_region
        ),
        model=model,
    )

    df_aggregated = get_aggregate_df(df, world_region=world_region)
    extras_idx = pd.MultiIndex.from_tuples(
        extra_variable_regions, names=["variable", "region"]
    )
    extras = pd.DataFrame(
        RNG.random((extras_idx.shape[0], df.columns.size)),
        columns=df.columns,
        index=extras_idx,
    )
    extras = set_new_single_value_levels(
        extras, {"model": model, "scenario": scenario, "unit": "Mt"}
    )

    df_to_check = pd.concat(
        [df_aggregated, extras.reorder_levels(df_aggregated.index.names)]
    )

    assert is_internally_consistent(df_to_check, model_regions=model_regions) is None


@pytest.mark.parametrize(
    "full_df, to_modify, model_regions",
    (
        pytest.param(
            full_df,
            to_modify,
            model_regions,
        )
        for model in ["model_a"]
        for world_region in ["World"]
        for model_regions in [[f"{model}|{r}" for r in ["Pacific OECD", "China"]]]
        for internal_consistency_checking_index in [
            get_internal_consistency_checking_index(
                model_regions, world_region=world_region
            )
        ]
        for full_df in [
            get_aggregate_df(
                get_df(internal_consistency_checking_index, model=model),
                world_region=world_region,
            )
        ]
        for to_modify in [
            *[[v] for v in internal_consistency_checking_index],
            # Get a selection of combos of dropped elements
            *[
                list(v)
                for v in list(
                    itertools.combinations(internal_consistency_checking_index, 2)
                )[:10]
            ],
        ]
    ),
)
def test_is_internally_consistent_incorrect_dataset(full_df, to_modify, model_regions):
    """
    Test that `is_internally_consistent` raises if the data is not internally consistent

    Here the internal inconsistency is because of an incorrrect sum
    """
    to_modify_index = pd.MultiIndex.from_tuples(to_modify, names=["variable", "region"])
    to_modify_locator = multi_index_match(full_df.index, to_modify_index)
    # + 1.1 as default CO2 tolerance is atol=1.0
    full_df.loc[to_modify_locator, :] += 1.1

    match = r"\s*.*".join([r"\s*".join([v, r]) for v, r in to_modify])
    with pytest.raises(InternalConsistencyError, match=match):
        is_internally_consistent(full_df, model_regions=model_regions)


@pytest.mark.parametrize(
    "full_df, to_remove, model_regions",
    (
        pytest.param(
            full_df,
            to_modify,
            model_regions,
        )
        for model in ["model_a"]
        for world_region in ["World"]
        for model_regions in [[f"{model}|{r}" for r in ["Pacific OECD", "China"]]]
        for internal_consistency_checking_index in [
            get_internal_consistency_checking_index(
                model_regions, world_region=world_region
            )
        ]
        for full_df in [
            get_aggregate_df(
                get_df(internal_consistency_checking_index, model=model),
                world_region=world_region,
            )
        ]
        for to_modify in [
            *[[v] for v in internal_consistency_checking_index],
            # Get a selection of combos of dropped elements
            *[
                list(v)
                for v in list(
                    itertools.combinations(internal_consistency_checking_index, 2)
                )[:10]
            ],
        ]
    ),
)
def test_is_internally_consistent_incorrect_dataset_missing_timeseries(
    full_df, to_remove, model_regions
):
    """
    Test that `is_internally_consistent` raises if the data is not internally consistent

    Here the internal inconsistency is because of missing reporting
    """
    to_remove_index = pd.MultiIndex.from_tuples(to_remove, names=["variable", "region"])
    full_df = full_df.loc[~multi_index_match(full_df.index, to_remove_index)]

    with pytest.raises(InternalConsistencyError):
        is_internally_consistent(
            full_df,
            model_regions=model_regions,
            # Have to make the tolerances tighter as our test data
            # is generated in the range [0, 1]
            tols={
                v: dict(rtol=1e-3, atol=1e-6)
                for v in (
                    "Emissions|BC",
                    "Emissions|CH4",
                    "Emissions|CO",
                    "Emissions|CO2",
                    "Emissions|NH3",
                    "Emissions|NOx",
                    "Emissions|OC",
                    "Emissions|Sulfur",
                    "Emissions|VOC",
                    "Emissions|N2O",
                )
            },
        )


@pytest.mark.parametrize(
    "model, extra_variable_regions",
    (
        (
            "model_a",
            (
                # Some random other sector
                ["Emissions|CO2|Other other", "model_a|China"],
                ["Emissions|CO2|Other other", "model_a|Pacific OECD"],
            ),
        ),
        (
            "model_a",
            (
                # Extra sector that is not used directly
                # but will influence the total
                ["Emissions|BC|Energy|Demand|Special", "model_a|China"],
                ["Emissions|BC|Energy|Demand|Special", "model_a|Pacific OECD"],
            ),
        ),
        (
            # Some extra sector only at the world level
            "model_a",
            (["Emissions|CH4|Unused", "World"],),
        ),
        (
            "model_a",
            (
                # Both of the above
                ["Emissions|N2O|Other other", "model_a|China"],
                ["Emissions|N2O|Other other", "model_a|Pacific OECD"],
                ["Emissions|NOx|Unused", "World"],
            ),
        ),
    ),
)
def test_is_internally_consistent_incorrect_dataset_extra_timeseries(
    model, extra_variable_regions
):
    """
    Check that `is_internally_consistent` raises that break the expected hierarchy
    with variables that aren't actually used for the gridding.

    In other words, check that we check what we're interested in,
    not the internal consistency of the entire dataset which is undefined
    because the internal consistency rules are undefined.
    """
    model_regions = [f"{model}|{r}" for r in ["Pacific OECD", "China"]]
    world_region = "World"

    extras_idx = pd.MultiIndex.from_tuples(
        extra_variable_regions, names=["variable", "region"]
    )
    base_index = get_internal_consistency_checking_index(
        model_regions, world_region=world_region
    ).append(extras_idx)

    # Aggregate up, including the extras.
    # The total will therefore rely on the extras,
    # but they won't be used when checking,
    # hence the error
    # (this is an error of alignment with what we care about, not within the hierarchy).
    df = get_aggregate_df(
        get_df(base_index=base_index, model=model), world_region=world_region
    )

    match = r"\s*.*".join([r"\s*".join([v, r]) for v, r in extra_variable_regions])
    with pytest.raises(InternalConsistencyError, match=match):
        is_internally_consistent(
            df,
            model_regions=model_regions,
            # Have to make the tolerances tighter as our test data
            # is generated in the range [0, 1]
            tols={
                v: dict(rtol=1e-6, atol=1e-8)
                for v in (
                    "Emissions|BC",
                    "Emissions|CH4",
                    "Emissions|CO",
                    "Emissions|CO2",
                    "Emissions|NH3",
                    "Emissions|NOx",
                    "Emissions|OC",
                    "Emissions|Sulfur",
                    "Emissions|VOC",
                    "Emissions|N2O",
                )
            },
        )


def test_to_complete_from_full_dataset():
    # start with the full index
    # call to complete
    # should get back the input
    # assert res.assumed_zero is None
    assert False


def test_to_complete_extra_timeseries():
    # Should get just the complete bits back i.e. extras are dropped
    # assert res.assumed_zero is None
    assert False


def test_to_complete_missing_timeseries():
    # Should get back a complete index with the optional timeseries as zeros
    # assert res.assumed_zero is equal to the missing timeseries filled with zeros
    assert False


def test_to_complete_extra_and_missing_optional_timeseries():
    # Should get just the complete bits back
    # Optional timeseries filled with zeros
    # assert res.assumed_zero is equal to the missing timeseries filled with zeros
    assert False
