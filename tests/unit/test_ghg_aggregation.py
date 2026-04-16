"""
Tests of the `gcages.ghg_aggregation` using ar6 data.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv

from gcages.ghg_aggregation import calculate_kyoto_ghg
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import (
    KEY_CMIP7_SCENARIOMIP_TESTING_MODEL_SCENARIOS,
    get_key_testing_model_scenario_parameters,
)

CMIP7_SCENARIOMIP_OUT_DIR = (
    Path(__file__).parents[1]
    / "regression/cmip7-scenariomip"
    / "cmip7-scenariomip-output"
)


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

    infilled = update_index_levels_func(
        infilled,
        {
            "variable": partial(
                convert_variable_name,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
    )

    res = calculate_kyoto_ghg(infilled)

    assert np.isclose(
        res[2100].values, exp.get(model), atol=1e-8
    ), f"Values don't match: {res[2100].values} vs {exp.get(model)}"
