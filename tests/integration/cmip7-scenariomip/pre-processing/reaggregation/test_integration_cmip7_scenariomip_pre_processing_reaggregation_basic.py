"""
Tests of basic reaggregation

'Basic' here means reaggregation assuming that domestic aviation
is reported at the model region level.
There may be other reaggregation methods we need to support,
hence why this is given a specific name.

There is lots of code in here,
because just generating example test cases
is too complex to be done by hand.
We deliberately don't use the code in `src`
for this so we can more easily ensure
that existing tests break as we make updates to our logic.
Yes, that means we have to change things in two places
when we make such updates,
but that's the point:
we get a clear indication that the change we made had the intended effect.
"""

from __future__ import annotations

import itertools
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

from gcages.aggregation import aggregate_df_level
from gcages.cmip7_scenariomip.gridding_emissions import get_complete_gridding_index
from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    assert_has_all_required_timeseries,
    to_complete,
    to_gridding_sectors,
)
from gcages.completeness import NotCompleteError, assert_all_groups_are_complete
from gcages.index_manipulation import (
    combine_sectors,
    combine_species,
    create_levels_based_on_existing,
    set_new_single_value_levels,
    split_sectors,
)
from gcages.testing import assert_frame_equal
from gcages.typing import NP_ARRAY_OF_FLOAT_OR_INT

# Required for assert_frame_equal
pytest.importorskip("pandas_indexing")

try:
    import openscm_units

    Q = openscm_units.unit_registry.Quantity
except ImportError:
    Q = None

pytestmark = pytest.mark.slow
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
        if self.gridding_sector in ["BECCS", "Other non-Land CDR"]:
            return tuple(
                f"Carbon Removal|CO2|{sector}" for sector in self.input_sectors
            )
        else:
            return tuple(
                f"Emissions|{species}|{sector}"
                for species in all_species
                for sector in self.input_sectors
            )

    def to_required_variables(self, all_species: tuple[str, ...]) -> tuple[str, ...]:
        if self.gridding_sector in ["BECCS", "Other non-Land CDR"]:
            return tuple(
                f"Carbon Removal|CO2|{sector}"
                for sector in self.input_sectors
                if sector not in self.input_sectors_optional
            )
        else:
            return tuple(
                f"Emissions|{species}|{sector}"
                for species in all_species
                for sector in self.input_sectors
                if not (
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
            input_sectors=(
                "AFOLU|Agriculture",
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
                "CO2",
            ),
        ),
        GriddingSectorComponentsReporting(
            gridding_sector="Agricultural Waste Burning",
            spatial_resolution="model region",
            input_sectors=("AFOLU|Agricultural Waste Burning",),
            input_sectors_optional=(),
            input_species_optional=("CO2",),
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
            input_species_optional=("CO2",),
        ),
        GriddingSectorComponentsReporting(
            gridding_sector="Grassland Burning",
            spatial_resolution="model region",
            input_sectors=("AFOLU|Land|Fires|Grassland Burning",),
            input_sectors_optional=(),
            input_species_optional=("CO2",),
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
        GriddingSectorComponentsReporting(
            gridding_sector="CO2 AFOLU",
            spatial_resolution="model region",
            input_sectors=("AFOLU",),
            input_sectors_optional=(),
            input_species_optional=(
                "BC",
                "CH4",
                "CO",
                "NH3",
                "N2O",
                "NOx",
                "OC",
                "Sulfur",
                "VOC",
            ),
        ),
        GriddingSectorComponentsReporting(
            gridding_sector="BECCS",
            spatial_resolution="model region",
            input_sectors=("Geological Storage|Biomass",),
            input_sectors_optional=(),
            input_species_optional=(
                "BC",
                "CH4",
                "CO",
                "NH3",
                "N2O",
                "NOx",
                "OC",
                "Sulfur",
                "VOC",
            ),
        ),
        GriddingSectorComponentsReporting(
            gridding_sector="Other non-Land CDR",
            spatial_resolution="model region",
            input_sectors=(
                "Ocean",
                "Geological Storage|Direct Air Capture",
                "Geological Storage|Synthetic Fuels",
                "Geological Storage|Other Sources",
                # "Long-Lived Materials",
                "Enhanced Weathering",
                # "Other",
            ),
            input_sectors_optional=(),
            input_species_optional=(
                "BC",
                "CH4",
                "CO",
                "NH3",
                "N2O",
                "NOx",
                "OC",
                "Sulfur",
                "VOC",
            ),
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

    # Forcing the AFOLU line be the sum of the AFOLU sectors
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
    for region in MODEL_REGIONS:
        for species in COMPLETE_GRIDDING_SPECIES:
            unit = unit_map[species]

            # Filter index to match only relevant sub-variables
            rows_to_sum = res.loc[
                res.index.get_level_values("variable").str.startswith(
                    f"Emissions|{species}|AFOLU|"
                )
                & (res.index.get_level_values("region") == region)
                & (res.index.get_level_values("unit") == unit)
                & (res.index.get_level_values("model") == model)
                & (res.index.get_level_values("scenario") == scenario)
            ]

            target_row = (
                f"Emissions|{species}|AFOLU",
                region,
                unit,
                "model_a",
                "scenario_a",
            )
            # Sum the rows
            if target_row in res.index and not rows_to_sum.empty:
                summed = rows_to_sum.sum(axis=0)
                res.loc[target_row] = summed

    # Keep Carbon Removal CO2 only
    mask = ~(
        res.index.get_level_values("variable").str.startswith("Carbon Removal|")
        & ~res.index.get_level_values("variable").str.startswith("Carbon Removal|CO2")
    )
    res = res[mask]

    return res


COMPLETE_DF = get_df(
    base_index=COMPLETE_INDEX,
    model=MODEL,
)

tuples_to_multi_index_vr = partial(
    pd.MultiIndex.from_tuples, names=["variable", "region"]
)


# =================
# Tests of `assert_has_all_required_timeseries`
# =================


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
def test_assert_has_all_required_timeseries(to_remove, to_add, exp):
    """
    Tests of `assert_has_all_required_timeseries`

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
        assert_has_all_required_timeseries(to_check, model_regions=model_regions)


def get_aggregate_df(
    indf: pd.DataFrame, world_region: str, on_clash_variable: str = "overwrite"
) -> pd.DataFrame:
    # The way the data is specified, a blind sum over regions is fine.
    df_regional = indf.loc[indf.index.get_level_values("region") != world_region]
    df_region_sum = groupby_except(indf, "region").sum()

    df_region_sum_aggregated = set_new_single_value_levels(
        aggregate_df_level(df_region_sum, level="variable", on_clash=on_clash_variable),
        {"region": world_region},
    )
    res = pd.concat(
        [df_regional, df_region_sum_aggregated.reorder_levels(df_regional.index.names)]
    )

    return res


# =================
# Tests of `to_complete`
# =================


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


# =================
# Tests of `to_gridding_sectors`
# =================


@pytest.fixture
def complete_to_gridding_res():
    variables = tuple(
        f"Emissions|{species}|Energy|Demand|Transportation"
        for species in COMPLETE_GRIDDING_SPECIES
    )
    regions = MODEL_REGIONS
    transport_index = pd.MultiIndex.from_product(
        [variables, regions], names=["variable", "region"]
    )

    unaggregated = get_df(COMPLETE_INDEX.difference(transport_index), model=MODEL)

    internally_consistent = get_aggregate_df(unaggregated, world_region=WORLD_REGION)

    # Add in the |Transportation level too
    transport = internally_consistent.loc[
        (
            internally_consistent.index.get_level_values("variable").str.contains(
                "Transportation"
            )
        )
        & (internally_consistent.index.get_level_values("region") != "World")
    ]
    transport = update_index_levels_func(
        transport, {"variable": lambda x: x.replace("|Domestic Aviation", "")}
    )

    internally_consistent = pd.concat(
        [
            internally_consistent,
            transport.reorder_levels(internally_consistent.index.names),
        ]
    )

    tcr = to_complete(internally_consistent, model_regions=MODEL_REGIONS)
    assert tcr.assumed_zero is None
    input = tcr.complete

    res = to_gridding_sectors(input)
    # The signs of the Carbon Removal are flipped when gridding. Below are restored to
    # the originals to allow comparisons.
    mask = res.index.get_level_values("variable").str.startswith("Carbon Removal")
    res.loc[mask] *= -1

    input_stacked = split_sectors(input).stack(future_stack=True).unstack("sectors")

    return internally_consistent, input_stacked, res


def test_complete_to_gridding_sectors_output_index(complete_to_gridding_res):
    _, _, res = complete_to_gridding_res

    complete_gridding_index = get_complete_gridding_index(model_regions=MODEL_REGIONS)
    assert_all_groups_are_complete(res, complete_gridding_index)
    # Make sure there are no extras
    assert res.shape[0] == complete_gridding_index.shape[0]


@pytest.mark.parametrize(
    "gridding_sector_definition",
    (
        pytest.param(gs, id=name)
        for name, gs in GRIDDING_SECTORS.items()
        if name
        not in (
            "Aircraft",
            "Domestic aviation headache",
            "Transportation Sector",
            "Energy Sector",
            "Industrial Sector",
        )
    ),
)
def test_complete_to_gridding_sectors_straightforward_sector(
    complete_to_gridding_res, gridding_sector_definition
):
    _, input_stacked, res = complete_to_gridding_res

    if gridding_sector_definition.spatial_resolution == "model region":
        region_locator = input_stacked.index.get_level_values("region") != WORLD_REGION

    else:
        region_locator = input_stacked.index.get_level_values("region") == WORLD_REGION

    tmp = input_stacked.loc[
        region_locator,
        list(gridding_sector_definition.input_sectors),
    ]
    # Separating CDR form Emissions
    if gridding_sector_definition.gridding_sector in ["BECCS", "Other non-Land CDR"]:
        tmp = tmp[tmp.index.get_level_values("table") != "Emissions"]
    else:
        tmp = tmp[tmp.index.get_level_values("table") != "Carbon Removal"]

    # Forcing CO2 dropped sectors to be 0 even in the CO2 case
    if gridding_sector_definition.gridding_sector in [
        "Agriculture",
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Peat Burning",
    ]:
        tmp.loc[tmp.index.get_level_values("species") == "CO2"] = 0.0

    if gridding_sector_definition.gridding_sector in ["CO2 AFOLU"]:
        tmp.loc[tmp.index.get_level_values("species") != "CO2"] = 0.0

    tmp[gridding_sector_definition.gridding_sector] = tmp.sum(axis="columns")
    exp = combine_sectors(
        tmp[[gridding_sector_definition.gridding_sector]]
        .unstack()
        .stack("sectors", future_stack=True)
    ).reorder_levels(res.index.names)

    assert_frame_equal(multi_index_lookup(res, exp.index), exp)


def test_complete_to_gridding_sectors_transport_and_aviation(complete_to_gridding_res):
    _, input_stacked, res = complete_to_gridding_res

    region_locator = (
        input_stacked.index.get_level_values("region") != WORLD_REGION
    ) & (input_stacked.index.get_level_values("table") != "Carbon Removal")

    domestic_aviation_sum = groupby_except(
        input_stacked.loc[region_locator][
            "Energy|Demand|Transportation|Domestic Aviation"
        ],
        "region",
    ).sum()
    tmp = input_stacked.loc[~region_locator].copy()
    tmp = tmp[tmp.index.get_level_values("table") != "Carbon Removal"]

    tmp["Aircraft"] = (
        tmp["Energy|Demand|Bunkers|International Aviation"] + domestic_aviation_sum
    )
    exp_aircraft = combine_sectors(
        tmp[["Aircraft"]].unstack().stack("sectors", future_stack=True)
    ).reorder_levels(res.index.names)

    assert_frame_equal(multi_index_lookup(res, exp_aircraft.index), exp_aircraft)

    tmp = input_stacked.loc[region_locator].copy()
    tmp["Transportation Sector"] = (
        tmp["Energy|Demand|Transportation"]
        - tmp["Energy|Demand|Transportation|Domestic Aviation"]
    )
    exp_transport = combine_sectors(
        tmp[["Transportation Sector"]].unstack().stack("sectors", future_stack=True)
    ).reorder_levels(res.index.names)

    assert_frame_equal(multi_index_lookup(res, exp_transport.index), exp_transport)


def test_complete_to_gridding_sectors_totals_preserved(complete_to_gridding_res):
    internally_consistent, _, res = complete_to_gridding_res

    res_totals = set_new_single_value_levels(
        combine_species(
            groupby_except(split_sectors(res), ["region", "sectors"]).sum()
        ),
        {"region": "World"},
    ).reorder_levels(internally_consistent.index.names)

    exp_totals = multi_index_lookup(internally_consistent, res_totals.index)

    # To account for CDR in the sum
    exp_totals.loc["Emissions|CO2"] = (
        exp_totals.loc["Emissions|CO2"].values
        + exp_totals.loc["Carbon Removal|CO2"].values
    )

    assert_frame_equal(res_totals, exp_totals)
