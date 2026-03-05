"""
Download CMIP7 ScenarioMIP global-workflow history and save it as CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_indexing  # noqa: F401
import pooch
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.renaming import SupportedNamingConventions, convert_variable_name


def main() -> None:
    URL = (
        "https://zenodo.org/records/17845154/files/"
        "global-workflow-history_202511261223_202511040855_202512032146_202512021030_"
        "7e32405ade790677a6022ff498395bff00d9792d_202511040855_202512071232_"
        "202511040855_202511040855_0002_0002.feather?download=1"
    )
    KNOWN_HASH = (
        "sha256:6bb21c3bc92bfaac7d93bc87c8d72b5e597656c6e84b6369179e54e0dbcaae77"
    )

    here = Path(__file__).resolve().parent
    output_dir = here / "cmip7-scenariomip-workflow-inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    feather_path = Path(
        pooch.retrieve(
            url=URL,
            known_hash=KNOWN_HASH,
            fname="history_cmip7_scenariomip.feather",
            path=output_dir,
            progressbar=True,
        )
    )
    csv_path = output_dir / "history_cmip7_scenariomip.csv"

    df = pd.read_feather(feather_path)

    # Convert names to gcages naming before saving
    df = update_index_levels_func(
        df,
        {
            "variable": lambda x: convert_variable_name(
                x,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
        copy=False,
    )

    df.to_csv(csv_path)


if __name__ == "__main__":
    main()
