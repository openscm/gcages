import numpy as np
import pandas as pd
import pytest

from gcages.cmip7_scenariomip.scm_running import get_complete_scenarios_for_magicc


def _multi_index_lookup(
    history: pd.DataFrame, scenario_index: pd.MultiIndex
) -> pd.DataFrame:
    return history.loc[scenario_index]


def test_get_complete_scenarios_for_magicc_adds_history_and_keeps_scenarios(
    monkeypatch,
):
    scenarios = pd.DataFrame(
        {
            2015: [10.0, 10.0],
            2016: [12.0, 14.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("M1", "S1", "CO2", "MtCO2/yr"),
                ("M1", "S1", "CH4", "MtCH4/yr"),
            ],
            names=["model", "scenario", "variable", "unit"],
        ),
    )

    history = pd.DataFrame(
        {
            2013: [6.0, 6.0],
            2014: [8.0, 8.0],
            2015: [10.0, 10.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("M1", "S1", "CO2", "MtCO2/yr"),
                ("M1", "S1", "CH4", "MtCH4/yr"),
            ],
            names=["model", "scenario", "variable", "unit"],
        ),
    )

    out = get_complete_scenarios_for_magicc(scenarios, history, magicc_start_year=2014)

    # columns: 2014, 2015, 2016
    assert list(out.columns) == [2014, 2015, 2016]
    assert out.loc[("M1", "S1", "CO2", "MtCO2/yr"), 2014] == 8.0
    assert out.loc[("M1", "S1", "CH4", "MtCH4/yr"), 2014] == 8.0
    assert out.loc[("M1", "S1", "CO2", "MtCO2/yr"), 2015] == 10.0
    assert out.loc[("M1", "S1", "CH4", "MtCH4/yr"), 2016] == 14.0


def test_get_complete_scenarios_for_magicc_interpolates_missing_years(monkeypatch):
    scenarios = pd.DataFrame(
        {
            2015: [10.0, 10.0],
            2016: [np.nan, np.nan],
            2017: [12.0, 14.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("M1", "S1", "CO2", "MtCO2/yr"),
                ("M1", "S1", "CH4", "MtCH4/yr"),
            ],
            names=["model", "scenario", "variable", "unit"],
        ),
    )

    history = pd.DataFrame(
        {
            2012: [6.0, 6.0],
            2013: [np.nan, np.nan],
            2014: [8.0, 8.0],
            2015: [10.0, 10.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("M1", "S1", "CO2", "MtCO2/yr"),
                ("M1", "S1", "CH4", "MtCH4/yr"),
            ],
            names=["model", "scenario", "variable", "unit"],
        ),
    )
    out = get_complete_scenarios_for_magicc(scenarios, history, magicc_start_year=2015)

    assert list(out.columns) == [2015, 2016, 2017]
    assert out.loc[("M1", "S1", "CO2", "MtCO2/yr"), 2016] == pytest.approx(11.0)
    assert out.loc[("M1", "S1", "CH4", "MtCH4/yr"), 2016] == 12.0
