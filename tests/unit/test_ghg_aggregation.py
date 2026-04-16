"""
Tests of the `gcages.ghg_aggregation` using ar6 data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.ghg_aggregation import calculate_kyoto_ghg
from gcages.renaming import SupportedNamingConventions
from gcages.testing import (
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS,
    get_key_testing_model_scenario_parameters,
)

CMIP7_SCENARIOMIP_OUT_DIR = (
    Path(__file__).parents[1]
    / "regression/cmip7-scenariomip"
    / "cmip7-scenariomip-output"
)

pytest.importorskip("openscm_units")


@get_key_testing_model_scenario_parameters(
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS
)
@pytest.mark.skip_ci_default
def test_ghg_kyoto(model, scenario):
    exp = {
        "REMIND-MAgPIE 3.5-4.11": -2117.6153824875214,
        "AIM 3.0": -15589.945051374172,
        "COFFEE 1.6": 480.653514665217,
        "GCAM 8s": 79997.47475468584,
        "IMAGE 3.4": 52572.39531797594,
        "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12": -1179.4559735383446,
        "WITCH 6.0": 6408.325974409462,
    }

    file = CMIP7_SCENARIOMIP_OUT_DIR / f"{model}_{scenario}_infilled.csv"
    infilled = load_timeseries_csv(
        file,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )

    tmp = infilled.xs("Emissions|CO2|AFOLU", level="variable") + infilled.xs(
        "Emissions|CO2|Energy and Industrial Processes", level="variable"
    )
    tmp["variable"] = "Emissions|CO2"
    tmp = tmp.set_index("variable", append=True)
    tmp = tmp.reorder_levels(["model", "scenario", "region", "variable", "unit"])
    infilled = pd.concat([infilled, tmp])

    res = calculate_kyoto_ghg(
        infilled,
        indf_naming_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    )
    assert np.isclose(
        res[2100].values, exp.get(model), atol=1e-8
    ), f"Values don't match: {res[2100].values} vs {exp.get(model)}"
