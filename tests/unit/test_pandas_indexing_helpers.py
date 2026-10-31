"""
Tests of `gcages.pandas_indexing_helpers`

Whether pandas_indexing has already been imported is global state,
so each test runs in a fresh Python process.
"""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("pandas_indexing")


def run_in_fresh_process(code: str) -> None:
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr


def test_accessors_not_registered_if_not_already_imported():
    run_in_fresh_process(
        """
        import pandas as pd

        from gcages.pandas_indexing_helpers import import_pandas_indexing

        pix = import_pandas_indexing("test")

        for cls in (pd.DataFrame, pd.Series, pd.Index):
            for name in ("pix", "idx"):
                assert not hasattr(cls, name), (cls, name)
                assert name not in cls._accessors, (cls, name)

        # The functions still work
        df = pd.DataFrame(
            [[1.0], [2.0]],
            index=pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")], names=["v", "r"]),
        )
        assert df.loc[pix.isin(v="a")].shape[0] == 1

        # Calling again gives the same module, still without side effects
        assert import_pandas_indexing("test") is pix
        assert not hasattr(pd.DataFrame, "pix")
        """
    )


def test_normal_import_afterwards_registers_accessors():
    run_in_fresh_process(
        """
        import pandas as pd

        from gcages.pandas_indexing_helpers import import_pandas_indexing

        import_pandas_indexing("test")
        assert not hasattr(pd.DataFrame, "pix")

        import pandas_indexing  # noqa: F401

        for cls in (pd.DataFrame, pd.Series, pd.Index):
            assert hasattr(cls, "pix"), cls
        """
    )


def test_accessors_retained_if_already_imported():
    run_in_fresh_process(
        """
        import pandas as pd
        import pandas_indexing

        from gcages.pandas_indexing_helpers import import_pandas_indexing

        assert import_pandas_indexing("test") is pandas_indexing

        for cls in (pd.DataFrame, pd.Series, pd.Index):
            for name in ("pix", "idx"):
                assert hasattr(cls, name), (cls, name)
        """
    )
