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
from pandas_openscm.indexing import multi_index_lookup, multi_index_match

from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    get_default_internal_conistency_checking_tolerances,
    has_all_required_timeseries,
    is_internally_consistent,
    to_complete,
    to_gridding_sectors,
)
from gcages.completeness import NotCompleteError
from gcages.index_manipulation import (
    combine_sectors,
    create_levels_based_on_existing,
    set_new_single_value_levels,
    split_sectors,
)
from gcages.internal_consistency import InternalConsistencyError
from gcages.testing import assert_frame_equal, compare_close
from gcages.typing import NP_ARRAY_OF_FLOAT_OR_INT

try:
    import openscm_units

    Q = openscm_units.unit_registry.Quantity
except ImportError:
    Q = None

RNG = np.random.default_rng()


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


GRIDDING_SECTORS = {
    gs.gridding_sector: gs
    for gs in (
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
}


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
    GRIDDING_SECTORS.values(),
    complete=True,
    all_species=COMPLETE_GRIDDING_SPECIES,
    world_region=WORLD_REGION,
    model_regions=MODEL_REGIONS,
)

REQUIRED_INDEX = to_index(
    GRIDDING_SECTORS.values(),
    complete=False,
    all_species=COMPLETE_GRIDDING_SPECIES,
    world_region=WORLD_REGION,
    model_regions=MODEL_REGIONS,
)

OPTIONAL_INDEX = COMPLETE_INDEX.difference(REQUIRED_INDEX)

INTERNAL_CONSISTENCY_DOUBLE_COUNTERS = to_index(
    [GRIDDING_SECTORS["Domestic aviation headache"]],
    complete=True,
    all_species=COMPLETE_GRIDDING_SPECIES,
    world_region=WORLD_REGION,
    model_regions=MODEL_REGIONS,
)

INTERNAL_CONSISTENCY_INDEX = COMPLETE_INDEX.difference(
    INTERNAL_CONSISTENCY_DOUBLE_COUNTERS
)
"""
To avoid double counting, you have to be careful

This is why the internal consistency index is its own thing.
"""


def guess_unit(v: str) -> str:
    species = v.split("|")[1]
    unit_map = {
        "BC": "Mt BC/yr",
        "CH4": "Mt CH4/yr",
        "CO": "Mt CO/yr",
        "CO2": "Gt C/yr",
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
    columns_name: str = "year",
) -> pd.DataFrame:
    res = set_new_single_value_levels(
        pd.DataFrame(
            RNG.random((base_index.shape[0], timepoints.size)),
            columns=pd.Index(timepoints, name=columns_name),
            index=create_levels_based_on_existing(
                base_index, create_from={"unit": ("variable", guess_unit)}
            ),
        ),
        {model_level: model, scenario_level: scenario},
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

    Beyond this, we rely on the implementation not having weird edge cases.
    """
    to_check = COMPLETE_DF
    model_regions = MODEL_REGIONS

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


def aggregate_df_level(indf: pd.DataFrame, on_clash: str = "raise") -> pd.DataFrame:
    # TODO: move this into pandas-openscm
    pytest.importorskip("pandas_indexing")

    level_separator = "|"
    level_to_aggregate = "variable"
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
            clash_locator = multi_index_match(
                indf_at_aggregated_level.index, to_aggregate_sum_combined.index
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
                    to_aggregate_sum_combined, indf_compare.index.remove_unused_levels()
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


def get_aggregate_df(
    indf: pd.DataFrame, world_region: str, on_clash_variable: str = "raise"
) -> pd.DataFrame:
    # The way the data is specified, a blind sum over regions is fine.
    df_regional = indf.loc[indf.index.get_level_values("region") != world_region]
    df_region_sum = groupby_except(indf, "region").sum()

    df_region_sum_aggregated = set_new_single_value_levels(
        aggregate_df_level(df_region_sum, on_clash=on_clash_variable),
        {"region": world_region},
    )
    res = pd.concat(
        [df_regional, df_region_sum_aggregated.reorder_levels(df_regional.index.names)]
    )

    return res


"""
The next few tests are of `is_internally_consistent`

The combinatorics of this are difficult
(you can't test all possible combinations of passing and missing things).

Our strategy is to make sure that:

1. an internally consistent dataset built off required timeseries,
   complete timeseries or things in between passes
1. modifications to an internally consistent set lead to the expected behaviour
    - modifications of the hierarchy we care about raise
    - modifications of other parts of the hierarchy cause no error
1. a dataset which is internally consistent according to its own hierarchy,
   but not according to the pieces we care about raises
"""

COMPLETE_INTERNALLY_CONSISTENT_DF = get_aggregate_df(
    get_df(INTERNAL_CONSISTENCY_INDEX, model=MODEL),
    world_region=WORLD_REGION,
)

SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF_STARTING_INDEX = (
    INTERNAL_CONSISTENCY_INDEX.difference(OPTIONAL_INDEX[::2])
)

NOT_IN_SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF_STARTING_INDEX = (
    INTERNAL_CONSISTENCY_INDEX.difference(
        SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF_STARTING_INDEX
    )
)

SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF = get_aggregate_df(
    get_df(
        SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF_STARTING_INDEX,
        model=MODEL,
    ),
    world_region=WORLD_REGION,
)


@pytest.mark.parametrize(
    "df",
    (
        pytest.param(
            COMPLETE_INTERNALLY_CONSISTENT_DF,
            id="complete",  # aggregation makes this complete
        ),
        pytest.param(
            get_aggregate_df(
                get_df(
                    INTERNAL_CONSISTENCY_INDEX.difference(OPTIONAL_INDEX), model=MODEL
                ),
                world_region=WORLD_REGION,
            ),
            id="required",
        ),
        pytest.param(
            SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF,
            id="in-between",
        ),
    ),
)
def test_is_internally_consistent_passes(df):
    assert (
        is_internally_consistent(
            df,
            model_regions=MODEL_REGIONS,
            tolerances=get_default_internal_conistency_checking_tolerances(),
        )
        is None
    )


def get_remove_modify_permutations(
    rows: pd.MultiIndex,
) -> Iterable[tuple[pd.MultiIndex | None, pd.MultiIndex | None, str]]:
    return (
        (rows, None, "removal"),
        (None, rows, "modification"),
    )


@pytest.mark.parametrize(
    "to_remove,to_modify,to_add,exp",
    (
        *(
            pytest.param(
                to_remove,
                to_modify,
                None,
                pytest.raises(InternalConsistencyError),
                id=f"internal-consistency-breaking-{id}",
            )
            for i in range(
                SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF_STARTING_INDEX.shape[0]
            )
            for to_remove, to_modify, id in get_remove_modify_permutations(
                SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF_STARTING_INDEX[[i]]
            )
        ),
        *(
            pytest.param(
                None,
                None,
                INTERNAL_CONSISTENCY_DOUBLE_COUNTERS[[i]],
                does_not_raise(),
                id=f"internal-consistency-non-breaking-{id}-as-double-counter",
            )
            for i in range(INTERNAL_CONSISTENCY_DOUBLE_COUNTERS.shape[0])
        ),
        *(
            pytest.param(
                to_remove,
                to_modify,
                None,
                pytest.raises(InternalConsistencyError),
                id=f"internal-consistency-breaking-multiple-{id}s",
            )
            for to_remove, to_modify, id in get_remove_modify_permutations(
                SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF_STARTING_INDEX[:6:2]
            )
        ),
        *(
            pytest.param(
                None,
                None,
                NOT_IN_SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF_STARTING_INDEX[[i]],
                pytest.raises(InternalConsistencyError),
                id="internal-consistency-breaking-addition",
            )
            for i in range(
                NOT_IN_SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF_STARTING_INDEX.shape[
                    0
                ]
            )
        ),
        pytest.param(
            None,
            None,
            NOT_IN_SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF_STARTING_INDEX[:6:2],
            pytest.raises(InternalConsistencyError),
            id="internal-consistency-breaking-multiple-additions",
        ),
        pytest.param(
            None,
            None,
            tuples_to_multi_index_vr(
                [
                    (
                        "Emissions|CO2|Energy|Demand|Transportation|Rail",
                        "World",
                    )
                ]
            ),
            does_not_raise(),
            id="random-extra",
        ),
        pytest.param(
            None,
            None,
            tuples_to_multi_index_vr(
                [
                    (
                        "Emissions|CO2|Energy|Demand|Transportation|Rail",
                        "World",
                    ),
                    *(
                        (
                            "Emissions|CO2|Energy|Demand|Transportation|Rail",
                            r,
                        )
                        for r in MODEL_REGIONS
                    ),
                ]
            ),
            does_not_raise(),
            id="multiple-extras",
        ),
    ),
)
def test_is_internally_consistent_modifications(to_remove, to_modify, to_add, exp):
    """
    Tests of modifications to an internally consistent dataset

    The combinatorics of this are difficult
    (you can't test all possible combinations of missing things).
    Our strategy is to make sure that:

    1. removing/modifying any level fails (as the internal consistency is broken),
       except those that don't affect internal consistency (e.g. domestic aviation)
    1. removing/modifying multiple levels fails (as the internal consistency is broken)
    1. adding any optional level fails (as the internal consistency is broken)
    1. adding multiple optional levels fails (as the internal consistency is broken)
    1. adding any extra level passes
       (doesn't affect internal consistency we care about,
       even though it does affect internal consistency
       if you think the pipe is meant to separate internally consistent levels)
       (in other words, make sure we only check the internal consistency
       we care about)
    1. adding multiple extra levels passes
    """
    to_check = SOMEWHAT_COMPLETE_INTERNALLY_CONSISTENT_DF
    model_regions = MODEL_REGIONS

    if to_remove is not None:
        to_remove_locator = multi_index_match(to_check.index, to_remove)
        assert to_remove_locator.sum() > 0, "Test won't do anything"
        to_check = to_check.loc[~to_remove_locator]

    if to_modify is not None:
        to_check = to_check.copy()
        to_modify_locator = multi_index_match(to_check.index, to_modify)
        assert to_modify_locator.sum() > 0, "Test won't do anything"
        # Huge change, tolerance sensitivity tested elsewhere
        to_check.loc[to_modify_locator] *= 3.0

    if to_add is not None:
        already_included_locator = multi_index_match(to_check.index, to_add)
        assert already_included_locator.sum() == 0, "Test won't do anything"
        to_add_df = get_df(to_add, model=MODEL)

        to_check = pd.concat([to_check, to_add_df.reorder_levels(to_check.index.names)])

    with exp:
        is_internally_consistent(
            to_check,
            model_regions=model_regions,
            tolerances=get_default_internal_conistency_checking_tolerances(),
        )


def test_is_internally_consistent_hierarchy_consistent():
    """
    Test that a dataset that is internally consistent from a hierarchy point of view,
    but not in the way we care about raises
    """
    base_index = INTERNAL_CONSISTENCY_INDEX.append(
        tuples_to_multi_index_vr(
            [
                (
                    # Some other sector we've never heard of
                    "Emissions|CH4|EnergyZN",
                    WORLD_REGION,
                )
            ]
        )
    )

    start_df = get_df(base_index=base_index, model=MODEL)
    df = get_aggregate_df(start_df, world_region=WORLD_REGION)
    with pytest.raises(InternalConsistencyError):
        is_internally_consistent(
            df,
            model_regions=MODEL_REGIONS,
            tolerances=get_default_internal_conistency_checking_tolerances(),
        )


@pytest.mark.parametrize(
    "delta,tol_kwargs,exp",
    (
        pytest.param(
            0.01,
            dict(atol=1e-4),
            pytest.raises(InternalConsistencyError),
            id="atol-raises",
        ),
        pytest.param(
            1.0,
            dict(atol=Q(1.0, "MtCO2 / yr")),
            pytest.raises(InternalConsistencyError),
            id="atol-raises-pint",
            marks=pytest.mark.skipif(Q is None, reason="Missing openscm-units"),
        ),
        pytest.param(
            0.01,
            dict(atol=1e-2),
            does_not_raise(),
            id="atol-edge-passes",
        ),
        pytest.param(
            0.010001,
            dict(atol=1e-2),
            pytest.raises(InternalConsistencyError),
            id="atol-overedge-raises",
        ),
        pytest.param(
            0.1,
            dict(atol=0.2),
            does_not_raise(),
            id="atol-passes",
        ),
        pytest.param(
            0.01,
            dict(rtol=1e-6),
            pytest.raises(InternalConsistencyError),
            id="rtol-raises",
        ),
        pytest.param(
            0.01,
            dict(rtol=Q(1e-6, "1")),
            pytest.raises(InternalConsistencyError),
            id="rtol-raises-pint",
            marks=pytest.mark.skipif(Q is None, reason="Missing openscm-units"),
        ),
        pytest.param(
            0.01,
            dict(rtol=1e-1),
            does_not_raise(),
            id="rtol-passes",
        ),
    ),
)
def test_is_internally_consistent_tolerance(delta, tol_kwargs, exp):
    start = get_df(INTERNAL_CONSISTENCY_INDEX, model=MODEL)
    # Set all values to one
    start.loc[:, :] = 1.0
    to_check = get_aggregate_df(start, world_region=WORLD_REGION)

    # Break consistency by given amount
    to_modify_locator = np.where(
        to_check.index.get_level_values("variable").str.startswith(
            "Emissions|CO2|Energy|Demand|Bunkers|International Shipping"
        )
    )[0][0]
    to_check.iloc[to_modify_locator, :] += delta

    tols = get_default_internal_conistency_checking_tolerances() | {
        "Emissions|CO2": tol_kwargs
    }
    with exp:
        is_internally_consistent(to_check, model_regions=MODEL_REGIONS, tolerances=tols)


def test_to_complete_from_full_dataset():
    res = to_complete(COMPLETE_DF, model_regions=MODEL_REGIONS)

    assert_frame_equal(res.complete, COMPLETE_DF)
    assert res.assumed_zero is None


@pytest.mark.parametrize(
    "to_add",
    (
        get_df(
            tuples_to_multi_index_vr(
                [
                    ("Emissions|CO2|ZN", "World"),
                    *(
                        ("Emissions|CO2|Energy|Demand|ZN", f"{MODEL}|{r}")
                        for r in MODEL_REGIONS
                    ),
                ]
            ),
            model=MODEL,
        ),
    ),
)
def test_to_complete_extra_timeseries(to_add):
    df = pd.concat(
        [
            COMPLETE_DF,
            to_add.reorder_levels(COMPLETE_DF.index.names),
        ]
    )
    res = to_complete(df, model_regions=MODEL_REGIONS)

    assert_frame_equal(res.complete, COMPLETE_DF)
    assert res.assumed_zero is None


@pytest.mark.parametrize(
    "to_remove",
    (
        *(pytest.param(OPTIONAL_INDEX[[i]]) for i in range(OPTIONAL_INDEX.shape[0])),
        pytest.param(OPTIONAL_INDEX[:4:2], id="multiple-levels-removed"),
        pytest.param(OPTIONAL_INDEX, id="required-only"),
    ),
)
def test_to_complete_missing_timeseries(to_remove):
    to_remove_locator = multi_index_match(COMPLETE_DF.index, to_remove)
    assert to_remove_locator.sum() > 0, "Test doing nothing"
    df = COMPLETE_DF.loc[~to_remove_locator]
    res = to_complete(df, model_regions=MODEL_REGIONS)

    exp_zeros = COMPLETE_DF.loc[to_remove_locator] * 0.0

    exp = pd.concat([df, exp_zeros.reorder_levels(df.index.names)])
    assert_frame_equal(res.complete, exp)
    assert_frame_equal(res.assumed_zero, exp_zeros)


@pytest.mark.parametrize(
    "to_remove",
    (
        *(pytest.param(OPTIONAL_INDEX[[i]]) for i in range(0, 12, 3)),
        pytest.param(OPTIONAL_INDEX[:4:2], id="multiple-levels-removed"),
        pytest.param(OPTIONAL_INDEX, id="required-only"),
    ),
)
@pytest.mark.parametrize(
    "to_add",
    (
        get_df(
            tuples_to_multi_index_vr([("Emissions|CO2|ZN", "World")]),
            model=MODEL,
        ),
        get_df(
            tuples_to_multi_index_vr(
                [
                    ("Emissions|CO2|ZN", "World"),
                    *(
                        ("Emissions|CO2|Energy|Demand|ZN", f"{MODEL}|{r}")
                        for r in MODEL_REGIONS
                    ),
                ]
            ),
            model=MODEL,
        ),
    ),
)
def test_to_complete_extra_and_missing_optional_timeseries(to_remove, to_add):
    to_remove_locator = multi_index_match(COMPLETE_DF.index, to_remove)
    assert to_remove_locator.sum() > 0, "Test doing nothing"
    df_after_removal = COMPLETE_DF.loc[~to_remove_locator]
    df = pd.concat(
        [
            df_after_removal,
            to_add.reorder_levels(COMPLETE_DF.index.names),
        ]
    )
    res = to_complete(df, model_regions=MODEL_REGIONS)

    exp_zeros = COMPLETE_DF.loc[to_remove_locator] * 0.0

    exp = pd.concat([df_after_removal, exp_zeros.reorder_levels(df.index.names)])
    assert_frame_equal(res.complete, exp)
    assert_frame_equal(res.assumed_zero, exp_zeros)


@pytest.fixture
def complete_to_gridding_res():
    input = COMPLETE_DF

    res = to_gridding_sectors(input)

    input_stacked = split_sectors(input).stack().unstack("sectors")

    return input_stacked, res


@pytest.mark.parametrize(
    "gridding_sector_definition",
    (
        pytest.param(gs, id=name)
        for name, gs in GRIDDING_SECTORS.items()
        if name
        not in ("Aircraft", "Domestic aviation headache", "Transportation Sector")
    ),
)
def test_complete_to_gridding_sectors_straightforward_sector(
    complete_to_gridding_res, gridding_sector_definition
):
    input_stacked, res = complete_to_gridding_res

    if gridding_sector_definition.spatial_resolution == "model region":
        region_locator = input_stacked.index.get_level_values("region") != WORLD_REGION

    else:
        region_locator = input_stacked.index.get_level_values("region") == WORLD_REGION

    tmp = input_stacked.loc[
        region_locator,
        list(gridding_sector_definition.input_sectors),
    ]
    tmp[gridding_sector_definition.gridding_sector] = tmp.sum(axis="columns")
    exp = combine_sectors(
        tmp[[gridding_sector_definition.gridding_sector]].unstack().stack("sectors")
    ).reorder_levels(res.index.names)

    assert_frame_equal(multi_index_lookup(res, exp.index), exp)


def test_complete_to_gridding_sectors_transport_and_aviation(complete_to_gridding_res):
    input, res = complete_to_gridding_res


def test_complete_to_gridding_sectors_totals_preserved(complete_to_gridding_res):
    input, res = complete_to_gridding_res


# Test of gridding sectors to global workflow go elsewhere
