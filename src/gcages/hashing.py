"""
Support for hash calculations
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def get_file_hash(
    fp: Path, algorithm: str = "sha256", buffer_size: int = 64 * 1024**2
) -> str:
    """
    Get the hash of a file

    Parameters
    ----------
    fp
        Path to the file

    algorithm
        Algorithm to use during hashing

    buffer_size
        The size of the chunks to use when hashing the file

    Returns
    -------
    :
        Hash of the file
    """
    hasher = hashlib.new(algorithm)
    with open(fp, "rb") as fh:
        data = fh.read(buffer_size)
        while data:
            hasher.update(data)
            data = fh.read(buffer_size)

    return hasher.hexdigest()


def get_data_checksum(df: pd.DataFrame) -> str:
    """
    Get a checksum of the data in a [pd.DataFrame][pandas.DataFrame]

    The checksum only depends on the index, columns and values,
    so it is independent of the order of the rows and columns
    and of how the data was serialised.
    This is useful for checking data that is saved to disk,
    because the bytes of a saved file can depend on the versions
    of the libraries used to write it
    (e.g. pandas embeds its version in feather files),
    so a file checksum would only match on the machine that generated it.

    Parameters
    ----------
    df
        Data of which to get the checksum

    Returns
    -------
    :
        sha256 checksum of the data
    """
    df_sorted = df.sort_index().sort_index(axis="columns")

    hasher = hashlib.sha256()
    hasher.update(repr(df_sorted.index.names).encode())
    for idx in df_sorted.index:
        hasher.update(repr(idx).encode())

    hasher.update(repr(df_sorted.columns.tolist()).encode())
    hasher.update(np.ascontiguousarray(df_sorted.to_numpy(dtype=np.float64)).tobytes())

    return hasher.hexdigest()
