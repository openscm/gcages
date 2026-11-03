# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gcages[cmip7scenariomip]",
#     "ixmp4>=0.17.0",
#     # pandas-indexing doesn't support pandas 3 yet
#     "pandas>=2.2.0,<3",
#     "pyarrow>=19.0.1",
# ]
#
# [tool.uv.sources]
# gcages = { path = "..", editable = true }
# ///
"""
Show that the workflow doesn't reproduce the published results from the latest data

We:

1. download the latest (i.e. default) version of each marker scenario
   from the scenariomip-internal ixmp4 platform
   (where the scenarios have their published names)
1. run them through the workflow
   ([CMIP7ScenarioMIPWorkflowRunner][gcages.workflows.cmip7_scenariomip.CMIP7ScenarioMIPWorkflowRunner])
1. check whether the infilled emissions match the published results
   (https://zenodo.org/records/19825038)

For some of the marker scenarios, the latest data on scenariomip-internal
is not the data that was used to create the published results,
so the workflow either fails or gives different results.
To see where the data differs,
run `cmip7-scenariomip-compare-pyam-ixmp4.py`.

Run with

```sh
uv run scripts/cmip7-scenariomip-ixmp4-latest.py
```

This requires an IIASA login with access to scenariomip-internal
(`uv run --with 'ixmp4>=0.17.0' ixmp4 login <your-username>`).
Downloads are cached, so data for a given version is only downloaded once.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import ixmp4
import pandas as pd

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
    Run the workflow with the latest data

    Returns
    -------
    :
        Exit code (0 if all the marker scenarios match the published results)
    """
    print("Setting up the workflow")
    runner = get_runner()
    published = load_published_emissions(downloader=PoochDownloader(cache_dir=DATA_DIR))

    results = {}
    for marker in CMIP7_SCENARIOMIP_MARKERS:
        print(f"Downloading the latest data for {marker.model} {marker.scenario}")
        raw, version = get_latest_marker_from_scenariomip_internal(marker)

        label = (
            f"{marker.model} {marker.published_scenario} "
            f"(scenariomip-internal v{version})"
        )
        print(f"Running the workflow for {label}")
        try:
            result = runner(raw)
        except Exception as exc:
            results[label] = f"The workflow failed: {type(exc).__name__}: {exc}"
            continue

        results[label] = check_against_published_emissions(
            result.infilled, published, marker=marker
        )

    print()
    exit_code = 0
    for label, problem in results.items():
        if problem is None:
            print(f"PASS: {label}")
        else:
            exit_code = 1
            print(f"FAIL: {label}\n{problem}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
