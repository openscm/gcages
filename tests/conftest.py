"""
Re-useable fixtures etc. for tests

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parents[1]

git = pytest.importorskip("git")
nomenclature = pytest.importorskip("nomenclature")


@pytest.fixture(scope="session", autouse=True)
def pandas_terminal_width():
    # Set pandas terminal width so that doctests don't depend on terminal width.

    # We set the display width to 120 because examples should be short,
    # anything more than this is too wide to read in the source.
    pd.set_option("display.width", 120)

    # Display as many columns as you want (i.e. let the display width do the
    # truncation)
    pd.set_option("display.max_columns", 1000)


@pytest.fixture(scope="session")
def default_data_structure_definition():
    exp_path = REPO_ROOT / "common-definitions"
    if not exp_path.exists():
        commit_id = "72707a466882b0ded4a582c27cc4d70f213215a7"
        repo_url = "https://github.com/IAMconsortium/common-definitions"
        msg = (
            f"Grabbing common-definitions, cloning to {exp_path} "
            f"and checking out commit {commit_id}. "
            "To update, delete the existing folder and re-run."
        )
        print(msg)
        repo = git.Repo.clone_from(repo_url, exp_path)
        repo.git.checkout(commit_id)

    dsd = nomenclature.definition.DataStructureDefinition(exp_path / "definitions")

    species_to_check = [
        "CO2",
        "CH4",
        "N2O",
        "BC",
        "CO",
        "NH3",
        "OC",
        "NOx",
        "Sulfur",
        "VOC",
    ]
    suffixes_to_check = [
        # Totals
        *species_to_check,
        # Energy
        # "Energy|Supply",
        # "Energy|Demand",
        # "Energy|Demand|Transportation",
        "Energy",
        "AFOLU",
        # "AFOLU|Land"
    ]
    for variable in dsd.variable:
        if (
            variable.startswith("Emissions")
            and any(species in variable for species in species_to_check)
            and any(variable.endswith(s) for s in suffixes_to_check)
        ):
            dsd.variable[variable].check_aggregate = True

    return dsd
