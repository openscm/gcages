"""
Download CMIP7 ScenarioMIP global-workflow history and save it as CSV.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    import pandas_indexing  # noqa: F401
    import pooch

    URL = "https://zenodo.org/records/17845154/files/global-workflow-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d_202511040855_202512071232_202511040855_202511040855_0002_0002.csv?download=1"
    KNOWN_HASH = "md5:19482df604f1dc746fb354ef66ef9047"

    here = Path(__file__).resolve().parent
    output_dir = here / "cmip7-scenariomip-workflow-inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    pooch.retrieve(
        url=URL,
        known_hash=KNOWN_HASH,
        fname="history_cmip7_scenariomip.csv",
        path=output_dir,
        progressbar=True,
    )


if __name__ == "__main__":
    main()
