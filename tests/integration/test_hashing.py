"""
Unit tests of `gcages.hashing`
"""

from __future__ import annotations

import hashlib

import pytest

from gcages.hashing import get_file_hash


@pytest.mark.parametrize(
    "algorithm, algorithm_exp", ((None, "sha256"), ("sha256", "sha256"), ("md5", "md5"))
)
def test_get_file_hash(algorithm, algorithm_exp):
    kwargs = {}
    if algorithm is not None:
        kwargs["algorithm"] = algorithm

    res = get_file_hash(__file__, **kwargs)

    hasher = hashlib.new(algorithm_exp)
    with open(__file__, "rb") as fh:
        hasher.update(fh.read())

    exp = hasher.hexdigest()

    assert res == exp
