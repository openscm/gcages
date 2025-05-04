"""
Tests of `gcages.harmonisation.aneris`
"""

import pandas as pd
import pytest

from gcages.assertions import MissingDataForTimesError
from gcages.harmonisation import assert_harmonised
from gcages.harmonisation.aneris import AnerisHarmoniser


@pytest.mark.parametrize("harmonisation_year", (2020.0, 2030.0))
def test_basic_aneris(harmonisation_year):
    pytest.importorskip("aneris")

    # Make sure things run without exploding
    historical_emissions = pd.DataFrame(
        [
            [1.0, 1.2, 1.5],
            [2.0, 1.2, 0.5],
        ],
        columns=[2010.0, 2020.0, 2030.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "r1", "GtCO2 / yr"),
                ("v2", "r1", "MtCH4 / yr"),
            ],
            names=["variable", "region", "unit"],
        ),
    )

    harmoniser = AnerisHarmoniser(
        historical_emissions=historical_emissions,
        harmonisation_year=harmonisation_year,
        progress=False,
        n_processes=None,
    )
    scenario_emissions = pd.DataFrame(
        [
            [1.3, 1.7, 1.0, 1.0],
            [1.0, 0.6, 0.5, 0.3],
        ],
        columns=[2020.0, 2030.0, 2050.0, 2100.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("s1", "m1", "v1", "r1", "GtCO2 / yr"),
                ("s1", "m1", "v2", "r1", "MtCH4 / yr"),
            ],
            names=["scenario", "model", "variable", "region", "unit"],
        ),
    )

    harmonised = harmoniser(scenario_emissions)

    assert_harmonised(
        harmonised,
        history=historical_emissions,
        harmonisation_time=harmonisation_year,
    )

    # Only data from the harmonisation year onwards is returned
    assert harmonised.columns[0] == harmonisation_year


def test_overrides_basic_aneris():
    pytest.importorskip("aneris")

    harmonisation_year = 2030.0

    historical_emissions = pd.DataFrame(
        [
            [1.0, 1.2, 1.5],
            [2.0, 1.2, 0.5],
        ],
        columns=[2010.0, 2020.0, 2030.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "r1", "GtCO2 / yr"),
                ("v2", "r1", "MtCH4 / yr"),
            ],
            names=["variable", "region", "unit"],
        ),
    )

    overrides = pd.Series(
        ["reduce_ratio_2050", "reduce_ratio_2100"],
        name="method",
        index=pd.MultiIndex.from_tuples(
            [
                ("v1",),
                ("v2",),
            ],
            names=["variable"],
        ),
    )

    harmoniser = AnerisHarmoniser(
        historical_emissions=historical_emissions,
        harmonisation_year=harmonisation_year,
        aneris_overrides=overrides,
        progress=False,
        n_processes=None,
    )

    scenario_emissions = pd.DataFrame(
        [
            [1.3, 1.7, 1.0, 0.9, 1.0],
            [1.0, 0.6, 0.5, 0.4, 0.3],
        ],
        columns=[2020.0, 2030.0, 2050.0, 2080.0, 2100.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("s1", "m1", "v1", "r1", "GtCO2 / yr"),
                ("s1", "m1", "v2", "r1", "MtCH4 / yr"),
            ],
            names=["scenario", "model", "variable", "region", "unit"],
        ),
    )

    harmonised = harmoniser(scenario_emissions)

    assert_harmonised(
        harmonised,
        history=historical_emissions,
        harmonisation_time=harmonisation_year,
    )

    # Check the overrides were used
    for locator, check_start in (
        [lambda x: x.index.get_level_values("variable") == "v1", 2050.0],
        [lambda x: x.index.get_level_values("variable") == "v2", 2100.0],
    ):
        pd.testing.assert_frame_equal(
            harmonised.loc[locator(harmonised), check_start:],
            scenario_emissions.loc[locator(scenario_emissions), check_start:],
        )


def test_overrides_multi_scenario_aneris():
    pytest.importorskip("aneris")

    harmonisation_year = 2020.0

    historical_emissions = pd.DataFrame(
        [
            [1.0, 1.2, 1.5],
            [2.0, 1.2, 0.5],
        ],
        columns=[2010.0, 2020.0, 2030.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "r1", "Gt / yr"),
                ("v2", "r1", "W / yr"),
            ],
            names=["variable", "region", "unit"],
        ),
    )

    overrides = pd.Series(
        [
            "reduce_ratio_2050",
            "reduce_ratio_2080",
            "reduce_ratio_2050",
            "reduce_ratio_2100",
        ],
        name="method",
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "s1"),
                ("v2", "s1"),
                ("v1", "s2"),
                ("v2", "s2"),
            ],
            names=["variable", "scenario"],
        ),
    )

    harmoniser = AnerisHarmoniser(
        historical_emissions=historical_emissions,
        harmonisation_year=harmonisation_year,
        aneris_overrides=overrides,
        progress=False,
        n_processes=None,
    )

    scenario_emissions = pd.DataFrame(
        [
            [1.3, 1.7, 1.0, 0.9, 1.0],
            [1.0, 0.6, 0.5, 0.4, 0.3],
            [1.0, 2.0, 2.0, 2.0, 2.4],
            [1.1, 0.8, 0.0, -0.4, -0.3],
        ],
        columns=[2020.0, 2030.0, 2050.0, 2080.0, 2100.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("s1", "m1", "v1", "r1", "Gt / yr"),
                ("s1", "m1", "v2", "r1", "W / yr"),
                ("s2", "m1", "v1", "r1", "Gt / yr"),
                ("s2", "m1", "v2", "r1", "W / yr"),
            ],
            names=["scenario", "model", "variable", "region", "unit"],
        ),
    )

    harmonised = harmoniser(scenario_emissions)

    assert_harmonised(
        harmonised,
        history=historical_emissions,
        harmonisation_time=harmonisation_year,
    )

    # Check the overrides were used
    for locator, check_start in (
        [
            lambda x: (x.index.get_level_values("variable") == "v1"),
            2050.0,
        ],
        [
            lambda x: (x.index.get_level_values("variable") == "v2")
            & (x.index.get_level_values("scenario") == "s1"),
            2080.0,
        ],
        [
            lambda x: (x.index.get_level_values("variable") == "v2")
            & (x.index.get_level_values("scenario") == "s2"),
            2100.0,
        ],
    ):
        pd.testing.assert_frame_equal(
            harmonised.loc[locator(harmonised), check_start:],
            scenario_emissions.loc[locator(scenario_emissions), check_start:],
            check_like=True,
        )


def test_overrides_multi_scenario_aneris_not_all_scenarios_specified():
    pytest.importorskip("aneris")

    harmonisation_year = 2020.0

    historical_emissions = pd.DataFrame(
        [
            [1.0, 1.2, 1.5],
            [2.0, 1.2, 0.5],
        ],
        columns=[2010.0, 2020.0, 2030.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "r1", "Gt / yr"),
                ("v2", "r1", "W / yr"),
            ],
            names=["variable", "region", "unit"],
        ),
    )

    overrides = pd.Series(
        [
            "reduce_ratio_2050",
            "reduce_ratio_2080",
            "reduce_ratio_2050",
            "reduce_ratio_2100",
        ],
        name="method",
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "s1"),
                ("v2", "s1"),
                ("v1", "s2"),
                ("v2", "s2"),
            ],
            names=["variable", "scenario"],
        ),
    )

    harmoniser = AnerisHarmoniser(
        historical_emissions=historical_emissions,
        harmonisation_year=harmonisation_year,
        aneris_overrides=overrides,
        progress=False,
        n_processes=None,
    )

    scenario_emissions = pd.DataFrame(
        [
            [1.3, 1.7, 1.0, 0.9, 1.0],
            [1.0, 0.6, 0.5, 0.4, 0.3],
            [1.0, 2.0, 2.0, 2.0, 2.4],
            [1.1, 0.8, 0.0, -0.4, -0.3],
            [1.0, 2.0, 2.0, 2.0, 2.4],
            [1.1, 0.8, 0.0, -0.4, -0.3],
        ],
        columns=[2020.0, 2030.0, 2050.0, 2080.0, 2100.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("s1", "m1", "v1", "r1", "Gt / yr"),
                ("s1", "m1", "v2", "r1", "W / yr"),
                ("s2", "m1", "v1", "r1", "Gt / yr"),
                ("s2", "m1", "v2", "r1", "W / yr"),
                ("s3", "m1", "v1", "r1", "Gt / yr"),
                ("s3", "m1", "v2", "r1", "W / yr"),
            ],
            names=["scenario", "model", "variable", "region", "unit"],
        ),
    )

    harmonised = harmoniser(scenario_emissions)

    assert_harmonised(
        harmonised,
        history=historical_emissions,
        harmonisation_time=harmonisation_year,
    )

    # Check the overrides were used
    for locator, check_start in (
        [
            lambda x: (x.index.get_level_values("variable") == "v1")
            & (x.index.get_level_values("scenario").isin(["s1", "s2"])),
            2050.0,
        ],
        [
            lambda x: (x.index.get_level_values("variable") == "v2")
            & (x.index.get_level_values("scenario") == "s1"),
            2080.0,
        ],
        [
            lambda x: (x.index.get_level_values("variable") == "v2")
            & (x.index.get_level_values("scenario") == "s2"),
            2100.0,
        ],
    ):
        pd.testing.assert_frame_equal(
            harmonised.loc[locator(harmonised), check_start:],
            scenario_emissions.loc[locator(scenario_emissions), check_start:],
            check_like=True,
        )

    # Check the overrides were not used for s3
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            harmonised.loc[
                harmonised.index.get_level_values("scenario") == "s2"
            ].reset_index("scenario", drop=True),
            harmonised.loc[
                harmonised.index.get_level_values("scenario") == "s3"
            ].reset_index("scenario", drop=True),
        )


@pytest.mark.xfail(reason="unit support not fully implemented")
def test_overrides_multi_scenario_multi_unit_aneris():
    pytest.importorskip("aneris")

    harmonisation_year = 2020.0

    historical_emissions = pd.DataFrame(
        [
            [1.0, 1.2, 1.5],
            [2.0, 1.2, 0.5],
        ],
        columns=[2010.0, 2020.0, 2030.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "r1", "Gt / yr"),
                ("v2", "r1", "W / yr"),
            ],
            names=["variable", "region", "unit"],
        ),
    )

    overrides = pd.Series(
        [
            "reduce_ratio_2050",
            "reduce_ratio_2080",
            "reduce_ratio_2050",
            "reduce_ratio_2100",
        ],
        name="method",
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "s1"),
                ("v2", "s1"),
                ("v1", "s2"),
                ("v2", "s2"),
            ],
            names=["variable", "scenario"],
        ),
    )

    harmoniser = AnerisHarmoniser(
        historical_emissions=historical_emissions,
        harmonisation_year=harmonisation_year,
        aneris_overrides=overrides,
        progress=False,
        n_processes=None,
    )

    scenario_emissions = pd.DataFrame(
        [
            [1.3, 1.7, 1.0, 0.9, 1.0],
            [1.0, 0.6, 0.5, 0.4, 0.3],
            [1010.0, 2020.0, 2010.0, 2050.0, 2043.0],
            [1.1, 0.8, 0.0, -0.4, -0.3],
        ],
        columns=[2020.0, 2030.0, 2050.0, 2080.0, 2100.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("s1", "m1", "v1", "r1", "Gt / yr"),
                ("s1", "m1", "v2", "r1", "W / yr"),
                ("s2", "m1", "v1", "r1", "Mt / yr"),
                ("s2", "m1", "v2", "r1", "W / yr"),
            ],
            names=["scenario", "model", "variable", "region", "unit"],
        ),
    )

    harmonised = harmoniser(scenario_emissions)

    assert_harmonised(
        harmonised,
        history=historical_emissions,
        harmonisation_time=harmonisation_year,
    )

    # Check the overrides were used
    for locator, check_start in (
        [
            lambda x: (x.index.get_level_values("variable") == "v1"),
            2050.0,
        ],
        [
            lambda x: (x.index.get_level_values("variable") == "v2")
            & (x.index.get_level_values("scenario") == "s1"),
            2080.0,
        ],
        [
            lambda x: (x.index.get_level_values("variable") == "v2")
            & (x.index.get_level_values("scenario") == "s2"),
            2100.0,
        ],
    ):
        pd.testing.assert_frame_equal(
            harmonised.loc[locator(harmonised), check_start:],
            scenario_emissions.loc[locator(scenario_emissions), check_start:],
            check_like=True,
        )


def test_historical_missing_timeseries_aneris():
    pytest.importorskip("aneris")

    historical_emissions = pd.DataFrame(
        [
            [1.0, 1.2, 1.5],
        ],
        columns=[2010.0, 2020.0, 2030.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "r1", "GtCO2 / yr"),
            ],
            names=["variable", "region", "unit"],
        ),
    )

    harmoniser = AnerisHarmoniser(
        historical_emissions=historical_emissions,
        harmonisation_year=2030.0,
        progress=False,
        n_processes=None,
    )

    scenario_emissions = pd.DataFrame(
        [
            [1.3, 1.7, 1.0, 1.0],
            [1.0, 0.6, 0.5, 0.3],
        ],
        columns=[2020.0, 2030.0, 2050.0, 2100.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("s1", "m1", "v1", "r1", "GtCO2 / yr"),
                ("s1", "m1", "v2", "r1", "MtCH4 / yr"),
            ],
            names=["scenario", "model", "variable", "region", "unit"],
        ),
    )

    with pytest.raises(
        ValueError, match="The input emissions contains values that aren't in history"
    ):
        harmoniser(scenario_emissions)


@pytest.mark.parametrize("harmonisation_year", (2020.0, 2030.0))
def test_historical_extra_timeseries_aneris(harmonisation_year):
    pytest.importorskip("aneris")

    historical_emissions = pd.DataFrame(
        [
            [1.0, 1.2, 1.5],
            [2.0, 1.2, 0.5],
            [0.1, 0.2, 0.5],
        ],
        columns=[2010.0, 2020.0, 2030.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "r1", "GtCO2 / yr"),
                ("v2", "r1", "MtCH4 / yr"),
                ("v3", "r1", "MtCH4 / yr"),
            ],
            names=["variable", "region", "unit"],
        ),
    )

    harmoniser = AnerisHarmoniser(
        historical_emissions=historical_emissions,
        harmonisation_year=harmonisation_year,
        progress=False,
        n_processes=None,
    )

    scenario_emissions = pd.DataFrame(
        [
            [1.3, 1.7, 1.0, 1.0],
            [1.0, 0.6, 0.5, 0.3],
        ],
        columns=[2020.0, 2030.0, 2050.0, 2100.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("s1", "m1", "v1", "r1", "GtCO2 / yr"),
                ("s1", "m1", "v2", "r1", "MtCH4 / yr"),
            ],
            names=["scenario", "model", "variable", "region", "unit"],
        ),
    )

    harmonised = harmoniser(scenario_emissions)

    assert_harmonised(
        harmonised,
        history=historical_emissions,
        harmonisation_time=harmonisation_year,
    )

    # Only data from the harmonisation year onwards is returned
    assert harmonised.columns[0] == harmonisation_year


def test_historical_timeseries_missing_harmonisation_year_error_aneris():
    pytest.importorskip("aneris")

    historical_emissions = pd.DataFrame(
        [
            [1.0, 1.2],
            [2.0, 1.2],
        ],
        columns=[2010.0, 2020.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "r1", "GtCO2 / yr"),
                ("v2", "r1", "MtCH4 / yr"),
            ],
            names=["variable", "region", "unit"],
        ),
    )

    with pytest.raises(MissingDataForTimesError):
        AnerisHarmoniser(
            historical_emissions=historical_emissions,
            harmonisation_year=2030.0,
            progress=False,
            n_processes=None,
        )


def test_scenario_timeseries_missing_harmonisation_year_error_aneris():
    pytest.importorskip("aneris")

    historical_emissions = pd.DataFrame(
        [
            [1.0, 1.2, 1.5],
            [2.0, 1.2, 0.5],
        ],
        columns=[2010.0, 2020.0, 2030.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "r1", "GtCO2 / yr"),
                ("v2", "r1", "MtCH4 / yr"),
            ],
            names=["variable", "region", "unit"],
        ),
    )

    harmoniser = AnerisHarmoniser(
        historical_emissions=historical_emissions,
        harmonisation_year=2010.0,
        progress=False,
        n_processes=None,
    )
    scenario_emissions = pd.DataFrame(
        [
            [1.3, 1.7, 1.0, 1.0],
            [1.0, 0.6, 0.5, 0.3],
        ],
        columns=[2020.0, 2030.0, 2050.0, 2100.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("s1", "m1", "v1", "r1", "GtCO2 / yr"),
                ("s1", "m1", "v2", "r1", "MtCH4 / yr"),
            ],
            names=["scenario", "model", "variable", "region", "unit"],
        ),
    )

    with pytest.raises(MissingDataForTimesError):
        harmoniser(scenario_emissions)
