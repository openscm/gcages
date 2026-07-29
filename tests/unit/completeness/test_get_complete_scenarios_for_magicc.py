"""
Tests of `gcages.cmip7_scenariomip.scm_running.get_complete_scenarios_for_magicc`
"""

import pandas as pd
import pytest

from gcages.cmip7_scenariomip.scm_running import get_complete_scenarios_for_magicc


def test_get_complete_scenarios_for_magicc_keeps_identical_values():
    scenario = pd.DataFrame(
        [
            [1.0, 2.0],
            [3.0, 2.0],
            [1.0, 2.0],
        ],
        columns=[2015, 2100],
        index=pd.MultiIndex.from_tuples(
            [
                ("model_1", "scenario_1", "World", "Emissions|BC", "Mt BC/yr"),
                (
                    "model_1",
                    "scenario_1",
                    "World",
                    "Emissions|CO2",
                    "Mt CO2/yr",
                ),
                ("model_1", "scenario_1", "World", "Emissions|CO", "Mt CO/yr"),
            ],
            names=["model", "scenario", "region", "variable", "unit"],
        ),
    )
    history = scenario[[2015]].copy()
    history[2012] = history[2015] * 0.8
    history[2010] = history[2015] * 0.7
    history = history.sort_index(axis=1)
    history = history[~history.index.duplicated(keep="first")].reset_index(
        ["model", "scenario"], drop=True
    )
    scenario_magicc = get_complete_scenarios_for_magicc(scenario, history, 2012)
    assert not scenario_magicc.isnull().any().any()


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.0],
                    [3.0, 2.0],
                    [1.0, 2.0],
                    [3.0, 2.0],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("model_1", "scenario_1", "World", "Emissions|BC", "Mt BC/yr"),
                        (
                            "model_1",
                            "scenario_1",
                            "World",
                            "Emissions|CO2",
                            "Mt CO2/yr",
                        ),
                        ("model_1", "scenario_1", "World", "Emissions|CO", "Mt CO/yr"),
                        ("model_1", "scenario_1", "World", "Emissions|BC", "Mt BC/yr"),
                    ],
                    names=["model", "scenario", "region", "variable", "unit"],
                ),
            ),
            id="same-unit",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.0],
                    [3.0, 2.0],
                    [1.0, 2.0],
                    [3.0, 2.0],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("model_1", "scenario_1", "World", "Emissions|BC", "Mt BC/yr"),
                        (
                            "model_1",
                            "scenario_1",
                            "World",
                            "Emissions|CO2",
                            "Mt CO2/yr",
                        ),
                        ("model_1", "scenario_1", "World", "Emissions|CO", "Mt CO/yr"),
                        ("model_1", "scenario_1", "World", "Emissions|BC", "kt BC/yr"),
                    ],
                    names=["model", "scenario", "region", "variable", "unit"],
                ),
            ),
            id="different-unit",
        ),
    ],
)
def test_get_complete_scenarios_for_magicc_raises_on_duplicate_trajectory(scenario):

    history = scenario[[2015]].copy()
    history[2012] = history[2015] * 0.8
    history[2010] = history[2015] * 0.7
    history = history.sort_index(axis=1)
    history = history[~history.index.duplicated(keep="first")].reset_index(
        ["model", "scenario"], drop=True
    )

    # Both scenarios repeat (model, scenario, variable): once with a matching unit
    # (same-unit) and once with a conflicting unit (different-unit). Either way
    # the (model, scenario, variable) key is duplicated, so it must raise.
    with pytest.raises(
        ValueError,
        match="'scenarios' has duplicate index: model, scenario, variable",
    ):
        get_complete_scenarios_for_magicc(scenario, history, 2015)
