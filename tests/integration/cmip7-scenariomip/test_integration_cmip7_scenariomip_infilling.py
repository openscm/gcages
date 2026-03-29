from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from gcages.cmip7_scenariomip import infilling


class TestInfillBranches:
    """Test all branches in core infilling functions."""

    def test_infill_nothing_to_infill_returns_none(self):
        """BRANCH: No variables need infilling → return None."""
        df = pd.DataFrame(
            {
                2015: [10.0, 10.0],
                2016: [12.0, 14.0],
            },
            index=pd.MultiIndex.from_tuples(
                [("M1", "S1", "Emissions|CH4"), ("M1", "S1", "Emissions|CO2")],
                names=["model", "scenario", "variable"],
            ),
        )
        infillers = {
            "Emissions|CH4": lambda x: x,
            "Emissions|CO2": lambda x: x,
        }

        result = infilling.infill(df, infillers)
        assert result is None

    def test_get_complete_indf_only(self):
        """BRANCH: infilled is None → return indf unchanged."""
        indf = pd.DataFrame(index=pd.MultiIndex.from_tuples([("M1", "S1", "CH4")]))
        result = infilling.get_complete(indf, None)
        pd.testing.assert_frame_equal(result, indf)

    def test_get_complete_both_indf_and_infilled(self):
        """BRANCH: Both indf and infilled → concat them."""
        indf = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([("M1", "S1", "Emissions|CH4")]),
            columns=[2015],
            data=[[10.0]],
        )
        infilled = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([("M1", "S1", "Emissions|N2O")]),
            columns=[2015],
            data=[[20.0]],
        )

        result = infilling.get_complete(indf, infilled)
        assert len(result) == 2
        assert result.loc[("M1", "S1", "Emissions|CH4"), 2015] == 10.0


class TestLoadFunctionsBranches:
    @patch("gcages.cmip7_scenariomip.infilling.get_file_hash")
    @patch("gcages.cmip7_scenariomip.infilling.load_timeseries_csv")
    def test_load_cmip7_scenariomip_infilling_db_check_hash_passes(
        self, mock_load, mock_hash
    ):
        """BRANCH: check_hash=True, hash matches → load success."""
        mock_hash.return_value = "3a55491330c0160a0c0abc011766559a"
        mock_load.return_value = pd.DataFrame()

        result = infilling.load_cmip7_scenariomip_infilling_db(
            Path("test.csv"), check_hash=True
        )
        mock_load.assert_called_once()
        assert result.empty

    @patch("gcages.cmip7_scenariomip.infilling.get_file_hash")
    def test_load_cmip7_scenariomip_infilling_db_hash_mismatch_raises(self, mock_hash):
        """BRANCH: check_hash=True, hash mismatch → AssertionError."""
        mock_hash.return_value = "wronghash"

        with pytest.raises(AssertionError, match="does not match"):
            infilling.load_cmip7_scenariomip_infilling_db(
                Path("test.csv"), check_hash=True
            )

    def test_load_cmip7_scenariomip_infilling_db_no_hash_check(self, monkeypatch):
        """BRANCH: check_hash=False → skip hash check."""
        monkeypatch.setattr(
            infilling, "load_timeseries_csv", lambda *a, **kw: pd.DataFrame()
        )
        monkeypatch.setattr(infilling, "get_file_hash", lambda *a, **kw: "ignored")

        result = infilling.load_cmip7_scenariomip_infilling_db(
            Path("test.csv"), check_hash=False
        )
        assert result.empty
