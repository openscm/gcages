"""
Download CMIP7 ScenarioMIP infilling database and save it as CSV.
"""

from __future__ import annotations

from pathlib import Path


# TODO: Not currently working.
# We believe this is because you have to be logged in to retrieve the file,
# and we haven't set that up
# (this should work fine once the record is no longer embargoed).
def main() -> None:
    import pandas_indexing  # noqa: F401
    import pooch

    URL = "https://zenodo.org/records/17844114/files/infiling-db_202512021030_202512071232_202511040855_202511040855.csv?download=1"
    KNOWN_HASH = "md5:3a55491330c0160a0c0abc011766559a"

    here = Path(__file__).resolve().parent
    output_dir = here / "cmip7-scenariomip-workflow-inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    pooch.retrieve(
        url=URL,
        known_hash=KNOWN_HASH,
        fname="infilling_db_cmip7_scenariomip.csv",
        path=output_dir,
        progressbar=True,
    )


if __name__ == "__main__":
    main()
