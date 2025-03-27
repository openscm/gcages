"""
Support for hash calculations
"""

from __future__ import annotations

import hashlib
from pathlib import Path


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
