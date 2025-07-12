"""
Check the change in a regression file

Helpful as the changes can be a bit misleading sometimes
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gcages.testing import compare_close

pd.options.display.max_rows = 500


def main() -> None:
    """
    Check the changes
    """
    base_commit = "7cd9b76"
    file_to_check = "/".join(
        (
            "tests",
            "regression",
            "cmip7-scenariomip",
            "test_regression_cmip7_scenariomip_pre_processing",
            "salted-202504-scenariomip-input_gridding_workflow_emissions.csv",
        )
    )
    rtol = 1e-4
    index = ["model", "scenario", "region", "variable", "unit"]

    base = (
        pd.read_csv(
            f"https://raw.githubusercontent.com/openscm/gcages/{base_commit}/{file_to_check}"
        )
        .set_index(index)
        .astype(float)
    )
    base.columns = base.columns.astype(int)

    current = pd.read_csv(Path(file_to_check)).set_index(index).astype(float)
    current.columns = current.columns.astype(int)

    # Only compare against common columns
    # because extra columns don't cause the regression test to fail
    common_cols = np.intersect1d(base.columns, current.columns)

    diffs = compare_close(
        left=base.loc[:, common_cols],
        left_name=base_commit,
        right=current.loc[:, common_cols],
        right_name="HEAD",
        rtol=rtol,
    ).unstack()
    print(diffs)
    print(sorted(diffs.index.get_level_values("variable").unique()))
    print(
        diffs.loc[
            diffs.index.get_level_values("variable").str.contains("Agricultural Waste"),
            :,
        ]
        .stack()
        .sort_index()
    )


if __name__ == "__main__":
    main()
