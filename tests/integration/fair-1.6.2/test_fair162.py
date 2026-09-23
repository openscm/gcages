"""
Unit test for FaIR 1.6.2
"""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pandas_openscm.index_manipulation import (
    update_index_levels_func,
)

from gcages.exceptions import MissingOptionalDependencyError
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running.fair import (
    FairSCMRunner,
    check_fair_version,
    load_fair_probabilistic_config,
)
from gcages.testing import get_ar6_infilled_emissions
from gcages.units_helpers import strip_pint_incompatible_characters_from_unit_string

# Only works if openscm-runner installed
pytest.importorskip("openscm_runner.adapters")

CONFIG_DIR = Path(__file__).parents[0] / "configs"
CFG_COMMON = CONFIG_DIR / "fair-1.6.2-wg3-params-common.json"
CFG_SLIM = CONFIG_DIR / "fair-1.6.2-wg3-params-slim.json"
PROCESSED_AR6_DB_DIR = (
    CONFIG_DIR.parents[2] / "regression" / "ar6" / "ar6-output-processed"
)


def strip_off_ar6_infilled_prefix_and_convert_to_gcages_and_fix_units(
    indf: pd.DataFrame,
) -> pd.DataFrame:
    indf = update_index_levels_func(
        indf,
        {
            "variable": lambda x: convert_variable_name(
                x.replace("AR6 climate diagnostics|Infilled|", ""),
                from_convention=SupportedNamingConventions.AR6_WG3,
                to_convention=SupportedNamingConventions.GCAGES,
            ),
            "unit": lambda x: strip_pint_incompatible_characters_from_unit_string(
                x
            ).replace("HFC245ca", "HFC245fa"),
        },
        copy=False,
    )

    return indf


def test_load_fair_probabilistic_config():

    with pytest.raises(
        NotImplementedError, match="Scenario end year set to 2111 must be < 2110"
    ):
        cfg = load_fair_probabilistic_config(
            config_file_slim=CFG_SLIM,
            config_file_common=CFG_COMMON,
            scenario_end_year=2111,
            num_cfgs=5,
        )

    cfg = load_fair_probabilistic_config(
        config_file_slim=CFG_SLIM,
        config_file_common=CFG_COMMON,
        scenario_end_year=2100,
        num_cfgs=5,
    )

    assert len(cfg) == 5
    first = cfg[0]
    assert len(first["natural"]) == 351
    assert len(first["F_solar"]) == 351
    assert len(first["F_volcanic"]) == 351
    assert len(first["scale"]) == 45
    assert len(first["C_pi"]) == 31
    assert len(first["E_pi"]) == 40
    assert len(first["b_aero"]) == 7

    slim = json.loads(CFG_SLIM.read_text())
    common = json.loads(CFG_COMMON.read_text())

    # CFC scaling factors are applied on top of the shared halogen scale
    assert first["scale"][15] == slim[0]["scale"][2] * common["cfc11_adj"]

    # pre-industrial concentrations
    assert first["C_pi"][0] == slim[0]["C_pi_CO2"]

    # aerosol params are padded with three zeros
    assert first["b_aero"] == [
        slim[0]["b_aero"][0],
        0.0,
        0.0,
        0.0,
        *slim[0]["b_aero"][1:4],
    ]


def test_check_fair_version(monkeypatch):
    # No version
    monkeypatch.setitem(sys.modules, "fair", None)
    with pytest.raises(MissingOptionalDependencyError):
        check_fair_version()

    # wrong version
    monkeypatch.setitem(sys.modules, "fair", SimpleNamespace(__version__="1.6.4"))
    with pytest.raises(AssertionError, match=re.escape("Expected fair v1.6.2.1")):
        check_fair_version()


def test_run_fair_162(dataframe_regression):
    pytest.importorskip("fair")
    pytest.importorskip("openscm_runner.adapters")

    complete = get_ar6_infilled_emissions(
        model="GCAM_5.3",
        scenario="NGFS2_Current_Policies",
        processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
    )
    # Select scenario and drop aggregated/cumulative rows
    is_aggregate = complete.index.get_level_values("variable").str.endswith(
        ("CO2", "F-Gases", "HFC", "PFC")
    ) | complete.index.get_level_values("variable").str.contains("Kyoto", regex=False)

    complete = complete[~is_aggregate]
    complete = strip_off_ar6_infilled_prefix_and_convert_to_gcages_and_fix_units(
        complete
    )

    end_year = int(complete.columns.max())

    runner = FairSCMRunner.load_configs(
        config_file_slim=CFG_SLIM,
        config_file_common=CFG_COMMON,
        scenario_end_year=end_year,
        num_cfgs=100,
        progress=False,
        output_variables=("Surface Air Temperature Change",),
    )
    res = runner(complete)

    dataframe_regression.check(res, default_tolerance=dict(rtol=1e-7))
