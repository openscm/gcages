# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gcages",
#     "ixmp4>=0.17.0",
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
Compare the data used in CMIP7 ScenarioMIP with the latest data

For each marker scenario, we compare:

- the version that was used in CMIP7 ScenarioMIP,
  downloaded from ssp_submission with pyam
  (with which the workflow reproduces the published results,
  see `cmip7-scenariomip-whole-process.py`)
- the latest version on the scenariomip-internal ixmp4 platform
  (with which the workflow doesn't reproduce the published results,
  see `cmip7-scenariomip-ixmp4-latest.py`)

We only compare the timeseries that are used in the workflow,
i.e. the timeseries used during pre-processing
plus the globally reported CO2 AFOLU emissions.

Run with

```sh
uv run scripts/cmip7-scenariomip-compare-pyam-ixmp4.py
```

This requires an IIASA login with access to both ssp_submission
and scenariomip-internal.
Downloads are cached, so data is only downloaded once.
"""

from __future__ import annotations

import functools
from pathlib import Path

import ixmp4
import pandas as pd
import pyam
import requests

from gcages.hashing import get_data_checksum
from gcages.workflows.cmip7_scenariomip import (
    CMIP7_SCENARIOMIP_MARKERS,
    CMIP7ScenarioMIPMarker,
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


SCENARIOMIP_INTERNAL_PLATFORM = "scenariomip-internal"
"""ixmp4 platform which holds the (renamed) ScenarioMIP scenarios"""


@functools.cache
def get_scenariomip_internal_platform() -> ixmp4.Platform:
    """
    Get the scenariomip-internal ixmp4 platform

    Cached so that we only connect (and authenticate) once.

    Returns
    -------
    :
        ixmp4 platform
    """
    return ixmp4.Platform(SCENARIOMIP_INTERNAL_PLATFORM)


def get_latest_marker_from_scenariomip_internal(
    marker: CMIP7ScenarioMIPMarker,
    out_dir: Path = DATA_DIR / "scenariomip-internal",
) -> tuple[pd.DataFrame, int]:
    """
    Get the latest (i.e. default) version of a marker from scenariomip-internal

    We only download the timeseries that are used in the workflow
    (see [get_used_variables][gcages.workflows.cmip7_scenariomip.pre_processing.get_used_variables]),
    to keep the load on the database as low as possible.

    The data is cached in `out_dir`.
    The data for a given version doesn't change,
    so the cache is used whenever there is a file for the default version.

    Parameters
    ----------
    marker
        Marker scenario

    out_dir
        Directory in which the data is cached

    Returns
    -------
    :
        Emissions for the marker (using the scenario name as submitted,
        i.e. `marker.scenario`, so it can be compared with the submitted data)
        and the version of the data on scenariomip-internal
    """  # noqa: E501
    platform = get_scenariomip_internal_platform()
    # Without a version, we get the default version
    run = platform.runs.get(marker.model, marker.published_scenario)

    fp = out_dir / get_filename(
        marker.model, marker.published_scenario, run.version, ext=".feather"
    )
    if fp.exists():
        res = pd.read_feather(fp)

    else:
        world_variables, model_region_variables = get_used_variables()
        res_l = [
            run.iamc.tabulate(
                variable={"name__in": list(world_variables)}, region={"name": "World"}
            ),
            run.iamc.tabulate(
                variable={"name__in": list(model_region_variables)},
                region={"name__like": f"{get_model_prefix(marker.model)}*"},
            ),
        ]
        # If nothing matches, ixmp4 returns a frame with no `year` column
        res_l = [v for v in res_l if not v.empty]
        if not res_l:
            msg = f"No data found for {marker=}"
            raise ValueError(msg)

        res_long = pd.concat(res_l).drop_duplicates()
        res_long["model"] = marker.model
        res_long["scenario"] = marker.published_scenario
        res = to_wide(res_long)
        res = res.loc[get_used_locator(res)]

        out_dir.mkdir(parents=True, exist_ok=True)
        res.to_feather(fp)

    res = res.rename(
        index={marker.published_scenario: marker.scenario}, level="scenario"
    )

    return res, run.version


RTOL = 1e-9
"""Relative tolerance to use when comparing values"""


def get_differences(pyam_df: pd.DataFrame, ixmp4_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the values which differ between the two sets of data

    Parameters
    ----------
    pyam_df
        Data from ssp_submission

    ixmp4_df
        Data from scenariomip-internal

    Returns
    -------
    :
        Values that differ, in long format
    """
    idx = pyam_df.index.intersection(ixmp4_df.index)
    years = sorted(
        set(pyam_df.dropna(how="all", axis="columns").columns)
        & set(ixmp4_df.dropna(how="all", axis="columns").columns)
    )
    a = pyam_df.loc[idx, years]
    b = ixmp4_df.loc[idx, years]

    abs_diff = (a - b).abs()
    rel_diff = abs_diff / b.abs().where(b != 0)
    differs = (rel_diff > RTOL) | ((b == 0) & (abs_diff > 0)) | (a.isna() != b.isna())

    differs_stacked = differs.stack()
    differs_idx = differs_stacked[differs_stacked].index

    res = differs_idx.to_frame(index=False)
    res["ssp_submission"] = a.stack(dropna=False).loc[differs_idx].to_numpy()
    res["scenariomip_internal"] = b.stack(dropna=False).loc[differs_idx].to_numpy()
    res["abs_diff"] = abs_diff.stack(dropna=False).loc[differs_idx].to_numpy()

    return res


def summarise(differences: pd.DataFrame) -> str:
    """
    Summarise differences by variable, region and year

    Parameters
    ----------
    differences
        Differences, as returned by `get_differences`

    Returns
    -------
    :
        Summary
    """
    is_world = differences["region"] == "World"
    lines = [
        f"{len(differences)} differing values in "
        f"{differences[['region', 'variable']].drop_duplicates().shape[0]} timeseries",
        f"World values that differ: {is_world.sum()}",
        "Other regions with differences: "
        f"{differences.loc[~is_world, 'region'].nunique()}",
        f"Years with differences: {sorted(differences['year'].unique().tolist())}",
        "",
        "By variable:",
        differences.groupby("variable")
        .agg(
            n_values=("year", "size"),
            n_regions=("region", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            max_abs_diff=("abs_diff", "max"),
            unit=("unit", "first"),
        )
        .sort_values("n_values", ascending=False)
        .to_string(),
        "",
        "Largest differences:",
        differences.sort_values("abs_diff", ascending=False)
        .head(10)
        .to_string(index=False),
    ]

    return "\n".join(lines)


def main() -> None:
    """
    Compare the data
    """
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 80)

    for marker in CMIP7_SCENARIOMIP_MARKERS:
        pyam_df = get_marker_from_ssp_submission(marker)
        ixmp4_df, ixmp4_version = get_latest_marker_from_scenariomip_internal(marker)

        pyam_used = pyam_df.loc[get_used_locator(pyam_df)]
        ixmp4_used = ixmp4_df.loc[get_used_locator(ixmp4_df)]

        print(
            f"==== {marker.model} {marker.scenario}: "
            f"ssp_submission v{marker.ssp_submission_version} vs "
            f"scenariomip-internal v{ixmp4_version} ({marker.published_scenario})"
        )
        only_pyam = pyam_used.index.difference(ixmp4_used.index)
        only_ixmp4 = ixmp4_used.index.difference(pyam_used.index)
        print(f"Used timeseries only in ssp_submission: {len(only_pyam)}")
        print(f"Used timeseries only in scenariomip-internal: {len(only_ixmp4)}")
        for name, only in (
            ("ssp_submission", only_pyam),
            ("scenariomip-internal", only_ixmp4),
        ):
            if not only.empty:
                print(f"Variables only in {name}:")
                print(only.to_frame(index=False)["variable"].value_counts().to_string())

        differences = get_differences(pyam_used, ixmp4_used)
        if differences.empty:
            print("All values in the used timeseries are the same")
        else:
            print(summarise(differences))

        print()


if __name__ == "__main__":
    main()
