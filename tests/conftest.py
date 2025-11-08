"""
Re-useable fixtures etc. for tests

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import pandas as pd
import pandas_openscm
import pytest


@pytest.fixture(scope="session", autouse=True)
def pandas_terminal_width():
    # Set pandas terminal width so that doctests don't depend on terminal width.

    # We set the display width to 120 because examples should be short,
    # anything more than this is too wide to read in the source.
    pd.set_option("display.width", 120)

    # Display as many columns as you want (i.e. let the display width do the
    # truncation)
    pd.set_option("display.max_columns", 1000)


@pytest.fixture()
def setup_pandas_accessors() -> None:
    # Not parallel safe, but good enough
    pandas_openscm.register_pandas_accessors()

    yield None

    # Surprising and a bit annoying that there isn't a safer way to do this
    pd.DataFrame._accessors.discard("openscm")
    if hasattr(pd.DataFrame, "openscm"):
        del pd.DataFrame.openscm

    pd.Series._accessors.discard("openscm")
    if hasattr(pd.Series, "openscm"):
        del pd.Series.openscm

    pd.Index._accessors.discard("openscm")
    if hasattr(pd.Index, "openscm"):
        del pd.Index.openscm
