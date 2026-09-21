"""
Unit test for FaIR 1.6.2
"""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.exceptions import MissingOptionalDependencyError
from gcages.renaming import SupportedNamingConventions, rename_variables
from gcages.scm_running.fair import (
    FairSCMRunner,
    check_fair_version,
    load_fair_probabilistic_config,
)

CONFIG_DIR = Path(__file__).parents[0] / "configs"
CFG_COMMON = CONFIG_DIR / "fair-1.6.2-wg3-params-common.json"
CFG_SLIM = CONFIG_DIR / "fair-1.6.2-wg3-params-slim.json"
CMIP7_SCENARIOMIP_OUT_DIR = (
    CONFIG_DIR.parents[2]
    / "regression"
    / "cmip7-scenariomip"
    / "cmip7-scenariomip-output"
)


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

    file = (
        CMIP7_SCENARIOMIP_OUT_DIR
        / "REMIND-MAgPIE 3.5-4.11_SSP1 - Very Low Emissions_complete.csv"
    )
    complete = load_timeseries_csv(
        file,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )
    # Select scenario and drop aggregated/cumulative rows
    is_aggregate = complete.index.get_level_values("variable").str.endswith(
        ("CO2", "F-Gases", "HFC", "PFC")
    ) | complete.index.get_level_values("variable").str.contains("Kyoto", regex=False)

    complete = complete[~is_aggregate]
    complete = rename_variables(
        complete,
        from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        to_convention=SupportedNamingConventions.GCAGES,
    )
    end_year = int(complete.columns.max())

    runner = FairSCMRunner.load_configs(
        config_file_slim=CFG_SLIM,
        config_file_common=CFG_COMMON,
        scenario_end_year=end_year,
        num_cfgs=5,
        progress=False,
        output_variables=("Surface Air Temperature Change",),
    )
    res = runner(complete)

    dataframe_regression.check(res, default_tolerance=dict(rtol=1e-7))
