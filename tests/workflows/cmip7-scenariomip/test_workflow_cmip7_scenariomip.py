"""
Test the CMIP7 ScenarioMIP workflow

We check that we reproduce v0.2 of the harmonised and infilled emissions
and climate assessment (MAGICC v7.6.0a3) for the CMIP7 ScenarioMIP marker scenarios
(https://zenodo.org/records/19825038).
As in CMIP7 ScenarioMIP, MAGICC is run with the emissions extended to 2500,
but the published results (and hence our checks) only go up to 2100.

The raw scenario data is in `inputs/<model>__<scenario>__v<version>.csv`,
created with `scripts/download-cmip7-scenariomip-inputs.py`.
The other inputs and the expected results are downloaded from Zenodo
into `inputs` when the tests run (these files are git ignored).
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pandas as pd
import pytest
from pandas_openscm.io import load_timeseries_csv

from gcages.testing import assert_frame_equal, guess_magicc_exe
from gcages.workflows.cmip7_scenariomip import (
    CMIP7_SCENARIOMIP_MARKERS,
    CMIP7ScenarioMIPWorkflowRunner,
    PoochDownloader,
    PublishedClimateAssessment,
    load_published_climate_assessment,
    load_published_emissions,
)

pix = pytest.importorskip("pandas_indexing")

INPUTS_DIR = Path(__file__).parent / "inputs"

DOWNLOADER = PoochDownloader(cache_dir=INPUTS_DIR)

MAGICC_DIR = (
    Path(__file__).parents[2]
    / "regression"
    / "cmip7-scenariomip"
    / "cmip7-scenariomip-workflow-inputs"
    / "magicc-v7.6.0a3"
)

CLIMATE_ATOL = 1e-5
"""
Absolute tolerance to use when comparing the climate assessment

The published values differ from ours at the level of ~1e-6
(K for temperatures, W/m^2 for effective radiative forcing),
presumably because the published results were created on a different platform.
"""

TIMESERIES_ATOL = 5e-3
"""
Absolute tolerance to use when comparing the climate assessment timeseries

The published results match a MAGICC run with emissions that end in 2100.
We run MAGICC with the emissions extended to 2500,
which changes the values in 2100
by up to ~2e-4 K for temperatures and ~2e-3 W/m^2 for effective radiative forcing
(MAGICC's value in a given year depends on the next year's emissions).
Before 2100, the differences are at the level of ~1e-5.
"""

METADATA_QUANTILE_ATOL = 5e-4
"""
Absolute tolerance (K) to use when comparing peak warming and warming in 2100

The published results match a MAGICC run with emissions that end in 2100.
We run MAGICC with the emissions extended to 2500,
which changes the warming in 2100 by up to ~2e-4 K
(MAGICC's value in a given year depends on the next year's emissions).
"""


@pytest.fixture(scope="module")
def runner() -> CMIP7ScenarioMIPWorkflowRunner:
    return CMIP7ScenarioMIPWorkflowRunner.from_cmip7_scenariomip_config(
        downloader=DOWNLOADER,
        magicc_exe_path=guess_magicc_exe(MAGICC_DIR / "bin"),
        magicc_prob_distribution_path=MAGICC_DIR
        / "configs"
        / "magicc-ar7-fast-track-drawnset-v0-3-0.json",
        progress=False,
        n_processes=multiprocessing.cpu_count(),
    )


@pytest.fixture(scope="module")
def published_emissions() -> pd.DataFrame:
    return load_published_emissions(downloader=DOWNLOADER)


@pytest.fixture(scope="module")
def published_climate_assessment() -> PublishedClimateAssessment:
    return load_published_climate_assessment(downloader=DOWNLOADER)


def load_raw_scenario(model: str, scenario: str, version: int) -> pd.DataFrame:
    """
    Load the raw scenario data

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    version
        Version of the model-scenario data in the ssp_submission database

    Returns
    -------
    :
        Raw scenario data
    """
    filename = f"{model}__{scenario}__v{version}.csv"
    filename = filename.replace("/", "_").replace(" ", "_")

    res = load_timeseries_csv(
        INPUTS_DIR / filename,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
        out_columns_name="year",
    )

    return res


@pytest.mark.magicc_v760a3
@pytest.mark.workflow_reproduction(workflow="cmip7-scenariomip")
@pytest.mark.parametrize(
    "model, scenario, version",
    [
        pytest.param(
            m.model,
            m.scenario,
            m.ssp_submission_version,
            id=f"{m.model}__{m.scenario}__v{m.ssp_submission_version}",
        )
        for m in CMIP7_SCENARIOMIP_MARKERS
    ],
)
def test_workflow_cmip7_scenariomip(  # noqa: PLR0913
    model,
    scenario,
    version,
    runner,
    published_emissions,
    published_climate_assessment,
):
    pytest.importorskip("aneris")
    pytest.importorskip("silicone")
    pytest.importorskip("openscm_runner")
    pytest.importorskip("openscm_units")

    raw = load_raw_scenario(model=model, scenario=scenario, version=version)

    result = runner(raw)

    exp = published_emissions.loc[pix.isin(model=model, scenario=scenario)]
    if exp.empty:
        raise AssertionError

    assert_frame_equal(
        result.infilled.loc[:, exp.columns].reorder_levels(exp.index.names),
        exp,
    )

    # Climate assessment (only up to 2100)
    published = published_climate_assessment
    res = result.post_processed

    exp_timeseries = published.timeseries_quantile.loc[
        pix.isin(model=model, scenario=scenario)
    ]
    if exp_timeseries.empty:
        raise AssertionError

    res_timeseries = res.timeseries_quantile.reorder_levels(
        exp_timeseries.index.names
    ).loc[exp_timeseries.index, exp_timeseries.columns]
    res_timeseries.columns.name = exp_timeseries.columns.name
    assert_frame_equal(res_timeseries, exp_timeseries, atol=TIMESERIES_ATOL)

    for res_metadata, exp_metadata, atol in (
        (res.metadata_quantile, published.metadata_quantile, METADATA_QUANTILE_ATOL),
        (
            res.metadata_exceedance_probabilities,
            published.metadata_exceedance_probabilities,
            CLIMATE_ATOL,
        ),
    ):
        exp_metadata_ms = exp_metadata.loc[pix.isin(model=model, scenario=scenario)]
        if exp_metadata_ms.empty:
            raise AssertionError

        pd.testing.assert_series_equal(
            res_metadata.reorder_levels(exp_metadata_ms.index.names).loc[
                exp_metadata_ms.index
            ],
            exp_metadata_ms,
            check_names=False,
            atol=atol,
        )
