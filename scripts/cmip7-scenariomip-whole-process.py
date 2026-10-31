# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gcages[cmip7scenariomip]",
#     # pandas-indexing doesn't support pandas 3 yet
#     "pandas>=2.2.0,<3",
#     "pyam-iamc>=3.4.0",
#     "pyarrow>=19.0.1",
#     "requests>=2.31.0",
# ]
#
# [tool.uv.sources]
# gcages = { path = "..", editable = true }
# ///
"""
Run the whole CMIP7 ScenarioMIP workflow, starting from the submitted data

We:

1. download the versions of the marker scenarios that were used in CMIP7 ScenarioMIP
   from the database to which they were submitted (ssp_submission, using pyam)
1. run them through the workflow
   ([CMIP7ScenarioMIPWorkflowRunner][gcages.workflows.cmip7_scenariomip.CMIP7ScenarioMIPWorkflowRunner])
1. check that the infilled emissions match the published results
   (https://zenodo.org/records/19825038)

Run with

```sh
uv run scripts/cmip7-scenariomip-whole-process.py
```

This requires an IIASA login with access to ssp_submission
(see `download-cmip7-scenariomip-inputs.py`).
Downloads are cached, so the script only downloads data the first time it is run.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import pandas as pd
import pyam
import requests

from gcages.hashing import get_data_checksum
from gcages.workflows.cmip7_scenariomip import (
    CMIP7_SCENARIOMIP_MARKERS,
    CMIP7ScenarioMIPMarker,
    CMIP7ScenarioMIPWorkflowRunner,
    PoochDownloader,
    load_published_emissions,
)
from gcages.workflows.cmip7_scenariomip.markers import (
    check_against_published_emissions,
)
from gcages.workflows.cmip7_scenariomip.pre_processing import (
    get_model_prefix,
    get_used_locator,
    get_used_variables,
)

DATA_DIR = (
    Path(__file__).parents[1] / "tests" / "workflows" / "cmip7-scenariomip" / "inputs"
)
"""
Directory in which to save data

This is where the tests look for their inputs,
so the scripts and the tests share downloads.
"""

INDEX_LEVELS = ("model", "scenario", "region", "variable", "unit")
"""Index levels of the emissions"""


def get_filename(model: str, scenario: str, version: int, ext: str) -> str:
    """
    Get the filename in which to save the data for a given run

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    version
        Version of the model-scenario data in the database

    ext
        File extension

    Returns
    -------
    :
        Filename
    """
    res = f"{model}__{scenario}__v{version}{ext}"
    res = res.replace("/", "_").replace(" ", "_")

    return res


def to_wide(res_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data from long format (one row per year) to wide (years as columns)

    Parameters
    ----------
    res_long
        Data in long format

    Returns
    -------
    :
        Data in wide format
    """
    res = res_long.set_index([*INDEX_LEVELS, "year"])["value"].unstack("year")
    res.columns = res.columns.astype(int)
    res.columns.name = "year"

    return res


SSP_SUBMISSION_CONNECTION = "ssp_submission"
"""pyam connection to the database to which the scenarios were submitted"""


@functools.cache
def get_ssp_submission_connection() -> pyam.iiasa.Connection:
    """
    Get the connection to the ssp_submission database

    Cached so that we only connect (and authenticate) once,
    and only if we actually need to download something.

    Returns
    -------
    :
        Connection to the database
    """
    return pyam.iiasa.Connection(SSP_SUBMISSION_CONNECTION)


@functools.cache
def get_ssp_submission_runs() -> pd.DataFrame:
    """
    Get information about all the runs in the ssp_submission database

    Returns
    -------
    :
        Information about all the runs (all versions, not only the default),
        including their run ID
    """
    return get_ssp_submission_connection().properties(default_only=False).reset_index()


def download_from_ssp_submission(marker: CMIP7ScenarioMIPMarker) -> pd.DataFrame:
    """
    Download the data used in the workflow for a marker from ssp_submission (via pyam)

    We only download the timeseries that are used in the workflow
    (see [get_used_variables][gcages.workflows.cmip7_scenariomip.pre_processing.get_used_variables]),
    to keep the load on the database as low as possible.

    Parameters
    ----------
    marker
        Marker for which to download the data
        (the version that was used in CMIP7 ScenarioMIP is downloaded)

    Returns
    -------
    :
        Emissions in wide format (years as columns)
    """  # noqa: E501
    model = marker.model
    scenario = marker.scenario
    version = marker.ssp_submission_version

    runs = get_ssp_submission_runs()
    run = runs[
        (runs["model"] == model)
        & (runs["scenario"] == scenario)
        & (runs["version"] == version)
    ]
    if run.shape[0] != 1:
        msg = (
            f"Expected exactly one run for {model=}, {scenario=}, {version=}, "
            f"found {run.shape[0]}"
        )
        raise ValueError(msg)

    conn = get_ssp_submission_connection()

    model_prefix = get_model_prefix(model)
    model_regions = [r for r in conn.regions() if r.startswith(model_prefix)]
    world_variables, model_region_variables = get_used_variables()

    # `pyam.iiasa.Connection.query` can only get the default version
    # or all versions of a model-scenario (which can be dozens),
    # so we query the API directly with the run ID instead
    # (this is the same request `pyam.iiasa.Connection.query` makes).
    res_l = []
    for regions, variables in (
        (["World"], world_variables),
        (model_regions, model_region_variables),
    ):
        response = requests.post(
            f"{conn._base_url}/runs/bulk/ts",
            headers={**conn.auth(), "Content-Type": "application/json"},
            json={
                "filters": {
                    "regions": list(regions),
                    "variables": list(variables),
                    "runs": [int(run["run_id"].iloc[0])],
                    "years": [],
                    "units": [],
                    "timeslices": [],
                }
            },
            timeout=600,
        )
        response.raise_for_status()
        res_l.append(pd.DataFrame(response.json()))

    res_long = pd.concat(res_l)
    if res_long.empty:
        msg = f"No data found for {model=}, {scenario=}, {version=}"
        raise ValueError(msg)

    if set(res_long["version"].unique()) != {version}:
        msg = f"Got data for other versions: {res_long['version'].unique()}"
        raise AssertionError(msg)

    if "subannual" in res_long and not res_long["subannual"].isin([-1, "Year"]).all():
        msg = "Got sub-annual data, which we don't support"
        raise AssertionError(msg)

    res = to_wide(res_long)

    return res.loc[get_used_locator(res)]


def get_marker_from_ssp_submission(
    marker: CMIP7ScenarioMIPMarker, out_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """
    Get the data used in the workflow for a marker, using the cache if possible

    If the file for the marker exists in `out_dir`
    and the checksum of its data matches the expected checksum
    (`marker.ssp_submission_data_checksum`), the data is loaded from disk.
    Otherwise, the data is downloaded,
    checked against the expected checksum and saved in `out_dir`.

    Parameters
    ----------
    marker
        Marker for which to get the data

    out_dir
        Directory in which the data is cached

    Returns
    -------
    :
        Emissions for the marker

    Raises
    ------
    ValueError
        The checksum of the downloaded data doesn't match the expected checksum
    """
    expected_checksum = marker.ssp_submission_data_checksum

    fp = out_dir / get_filename(
        marker.model, marker.scenario, marker.ssp_submission_version, ext=".feather"
    )
    if fp.exists():
        cached = pd.read_feather(fp)
        if get_data_checksum(cached) == expected_checksum:
            return cached

    res = download_from_ssp_submission(marker)

    checksum = get_data_checksum(res)
    if checksum != expected_checksum:
        msg = (
            f"Checksum of the data downloaded for {marker} "
            f"does not match the expected value. {checksum=} {expected_checksum=}"
        )
        raise ValueError(msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_feather(fp)

    return res


def get_runner() -> CMIP7ScenarioMIPWorkflowRunner:
    """
    Get the workflow runner

    Downloads from Zenodo are saved in `DATA_DIR`
    so they are shared with the tests.

    Returns
    -------
    :
        Workflow runner
    """
    return CMIP7ScenarioMIPWorkflowRunner.from_cmip7_scenariomip_config(
        downloader=PoochDownloader(cache_dir=DATA_DIR), progress=False
    )


def main() -> int:
    """
    Run the whole process

    Returns
    -------
    :
        Exit code (0 if all the marker scenarios match the published results)
    """
    print("Downloading the submitted data from ssp_submission")
    raw = pd.concat(
        [get_marker_from_ssp_submission(m) for m in CMIP7_SCENARIOMIP_MARKERS]
    )

    print("Setting up the workflow")
    runner = get_runner()

    print(f"Running the workflow for {len(CMIP7_SCENARIOMIP_MARKERS)} marker scenarios")
    result = runner(raw)

    print("Comparing with the published results")
    published = load_published_emissions(downloader=PoochDownloader(cache_dir=DATA_DIR))

    exit_code = 0
    for marker in CMIP7_SCENARIOMIP_MARKERS:
        problem = check_against_published_emissions(
            result.infilled, published, marker=marker
        )
        if problem is None:
            print(f"PASS: {marker.model} {marker.scenario}")
        else:
            exit_code = 1
            print(f"FAIL: {marker.model} {marker.scenario}\n{problem}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
