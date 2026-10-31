"""
Re-useable fixtures etc. for tests

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import random
from collections import defaultdict

import pandas as pd
import pytest

WORKFLOWS = ("cmip7-scenariomip",)
"""
Workflows for which we have reproduction tests

For details, see `docs/development/testing-strategy.md`.
"""

WORKFLOW_SELECTION_KEY = pytest.StashKey[dict]()
"""Key used to store the scenarios selected for each workflow"""


def pytest_addoption(parser):
    for workflow in WORKFLOWS:
        parser.addoption(
            f"--workflow-{workflow}",
            action="store_true",
            default=False,
            help=f"Run the reproduction tests for the {workflow} workflow",
        )

    parser.addoption(
        "--workflow-n-scenarios",
        type=int,
        default=2,
        help=(
            "Number of scenarios to run (picked at random) "
            "for each workflow whose reproduction tests are run"
        ),
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # trylast so that this runs after deselection by `-k` and `-m`.
    # As a result, if you select specific scenarios
    # (e.g. with `-k` or by node ID)
    # and there are no more than `--workflow-n-scenarios` of them,
    # they will all be run.
    workflow_items = defaultdict(list)
    for item in items:
        marker = item.get_closest_marker("workflow_reproduction")
        if marker is None:
            continue

        workflow = marker.kwargs["workflow"]
        if workflow not in WORKFLOWS:
            msg = f"Unknown workflow: {workflow!r}. Known workflows: {WORKFLOWS}"
            raise ValueError(msg)

        workflow_items[workflow].append(item)

    n_scenarios = config.getoption("--workflow-n-scenarios")
    selected = {}
    for workflow, w_items in workflow_items.items():
        if not config.getoption(f"--workflow-{workflow}"):
            skip = pytest.mark.skip(
                reason=f"Use --workflow-{workflow} to run the {workflow} workflow tests"
            )
            for item in w_items:
                item.add_marker(skip)

            continue

        if len(w_items) <= n_scenarios:
            selected[workflow] = w_items
            continue

        selected[workflow] = random.sample(w_items, n_scenarios)
        skip = pytest.mark.skip(
            reason=(
                f"Not one of the {n_scenarios} randomly selected scenarios "
                f"for the {workflow} workflow (see --workflow-n-scenarios)"
            )
        )
        for item in w_items:
            if item not in selected[workflow]:
                item.add_marker(skip)

    config.stash[WORKFLOW_SELECTION_KEY] = selected


def pytest_collection_finish(session):
    selected = session.config.stash.get(WORKFLOW_SELECTION_KEY, {})
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return

    for workflow, w_items in selected.items():
        reporter.write_line(f"Running {workflow} workflow tests for:")
        for item in w_items:
            reporter.write_line(f"  {item.nodeid}")


@pytest.fixture(scope="session", autouse=True)
def pandas_terminal_width():
    # Set pandas terminal width so that doctests don't depend on terminal width.

    # We set the display width to 120 because examples should be short,
    # anything more than this is too wide to read in the source.
    pd.set_option("display.width", 120)

    # Display as many columns as you want (i.e. let the display width do the
    # truncation)
    pd.set_option("display.max_columns", 1000)
