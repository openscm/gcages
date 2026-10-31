"""
Input data for the CMIP7 ScenarioMIP workflow

Large inputs are downloaded from Zenodo
(with a [Downloader][(m).], [PoochDownloader][(m).] by default).
Inputs that aren't available anywhere else are shipped with gcages.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Protocol

import pandas as pd
from attrs import define
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.cmip7_scenariomip.infilling import load_cmip7_scenariomip_ghg_inversions
from gcages.exceptions import MissingOptionalDependencyError
from gcages.hashing import get_file_hash
from gcages.renaming import SupportedNamingConventions, convert_variable_name


@define(frozen=True)
class ZenodoFile:
    """
    A file on Zenodo
    """

    record_id: str
    """ID of the Zenodo record the file belongs to"""

    filename: str
    """Name of the file"""

    md5: str
    """md5 hash of the file"""

    @property
    def url(self) -> str:
        """URL from which the file can be downloaded"""
        return (
            f"https://zenodo.org/records/{self.record_id}/files/{self.filename}"
            "?download=1"
        )


GLOBAL_HISTORY_FILE = ZenodoFile(
    record_id="17845154",
    filename="global-workflow-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d_202511040855_202512071232_202511040855_202511040855_0002_0002.parquet.gzip",
    md5="e85a93e11aed06645afb0b7d1b04f645",
)
"""Historical emissions used for harmonisation of the global-only emissions"""

REGION_SECTOR_HISTORY_FILE = ZenodoFile(
    record_id="17845154",
    filename="gridding-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d.parquet.gzip",
    md5="dfe5803810f69ca0753a77e1e8d45f9d",
)
"""Historical emissions used for harmonisation at the region-sector level"""

INFILLING_DB_FILE = ZenodoFile(
    record_id="17844114",
    filename="infiling-db_202512021030_202512071232_202511040855_202511040855.parquet.gzip",
    md5="541e06c42159fa7f59e202d8451c2f1f",
)
"""Infilling database"""


class Downloader(Protocol):
    """
    Downloader of files from Zenodo
    """

    def __call__(self, file: ZenodoFile) -> Path:
        """
        Download the file (if needed)

        Parameters
        ----------
        file
            File to download

        Returns
        -------
        :
            Path to the local copy of the file
        """


@define
class PoochDownloader:
    """
    Downloader which uses [pooch](https://www.fatiando.org/pooch/)

    Files are only downloaded if they aren't already in `cache_dir`
    (or if the copy in `cache_dir` doesn't have the expected hash).
    """

    cache_dir: Path | None = None
    """
    Directory in which to save the downloaded files

    If not supplied, we use pooch's default cache directory for gcages.
    """

    progressbar: bool = False
    """Should a progress bar be shown while downloading?"""

    def __call__(self, file: ZenodoFile) -> Path:
        """
        Download the file (if needed)

        Parameters
        ----------
        file
            File to download

        Returns
        -------
        :
            Path to the local copy of the file
        """
        try:
            import pooch  # type: ignore # pooch has no type hints # noqa: PLC0415
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "PoochDownloader", requirement="pooch"
            ) from exc

        cache_dir = (
            self.cache_dir if self.cache_dir is not None else pooch.os_cache("gcages")
        )

        return Path(
            pooch.retrieve(
                url=file.url,
                known_hash=f"md5:{file.md5}",
                fname=file.filename,
                path=cache_dir,
                progressbar=self.progressbar,
            )
        )


def get_file(file: ZenodoFile, downloader: Downloader) -> Path:
    """
    Get a file from Zenodo

    Parameters
    ----------
    file
        File to get

    downloader
        Downloader to use

    Returns
    -------
    :
        Path to the local copy of the file

    Raises
    ------
    AssertionError
        The file returned by `downloader` doesn't have the expected hash
    """
    res = downloader(file)

    # Check the hash ourselves,
    # so we don't rely on every downloader doing it
    fp_hash = get_file_hash(res, algorithm="md5")
    if fp_hash != file.md5:
        msg = (
            f"The md5 hash of {res} is {fp_hash}. "
            f"This does not match what we expect ({file.md5})."
        )
        raise AssertionError(msg)

    return res


def load_parquet(file: ZenodoFile, downloader: Downloader) -> pd.DataFrame:
    """
    Load a parquet file from Zenodo

    Parameters
    ----------
    file
        File to load

    downloader
        Downloader to use

    Returns
    -------
    :
        Loaded data
    """
    res = pd.read_parquet(get_file(file, downloader=downloader))
    res.columns.name = "year"

    return res


def convert_cmip7_scenariomip_names_to_gcages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert variable names from CMIP7 ScenarioMIP's convention to gcages'

    Parameters
    ----------
    df
        Data to convert

    Returns
    -------
    :
        Data with gcages variable names
    """
    return update_index_levels_func(
        df,
        {
            "variable": lambda x: convert_variable_name(
                x,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
    )


def get_shipped_file(filename: str) -> Path:
    """
    Get a file which is shipped with gcages

    Parameters
    ----------
    filename
        Name of the file

    Returns
    -------
    :
        Path to the file
    """
    return Path(
        importlib.resources.files("gcages.workflows.cmip7_scenariomip").joinpath(  # type: ignore # some pathlib bug (result should just be a path...)
            "data", filename
        )
    )


def load_cmip7_ghg_inversions() -> pd.DataFrame:
    """
    Load the CMIP7 GHG inversions (CR-CMIP-1-0-0) used for infilling

    Returns
    -------
    :
        CMIP7 GHG inversions, using gcages' naming convention
    """
    res = load_cmip7_scenariomip_ghg_inversions(
        get_shipped_file("cmip7_ghg_inversions.csv")
    )

    return update_index_levels_func(
        res,
        {
            "variable": lambda x: convert_variable_name(
                x,
                from_convention=SupportedNamingConventions.OPENSCM_RUNNER,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
    )


def load_harmonisation_methods(filename: str) -> pd.Series[str]:
    """
    Load harmonisation methods used in CMIP7 ScenarioMIP

    These are the methods that were used for the marker scenarios
    in v0.2 of the harmonised emissions (https://zenodo.org/records/19825038).

    Parameters
    ----------
    filename
        Name of the shipped file from which to load the methods

    Returns
    -------
    :
        Harmonisation methods
    """
    res = pd.read_csv(get_shipped_file(filename)).set_index(
        ["model", "scenario", "region", "variable"]
    )["method"]

    return res
