"""
Exceptions that are used throughout
"""

from __future__ import annotations

import difflib
from collections.abc import Collection
from typing import Any


class MissingOptionalDependencyError(ImportError):
    """
    Raised when an optional dependency is missing

    For example, plotting dependencies like matplotlib
    """

    def __init__(self, callable_name: str, requirement: str) -> None:
        """
        Initialise the error

        Parameters
        ----------
        callable_name
            The name of the callable that requires the dependency

        requirement
            The name of the requirement
        """
        error_msg = f"`{callable_name}` requires {requirement} to be installed"
        super().__init__(error_msg)


class UnrecognisedValueError(ValueError):
    """
    Raised when a value is not recognised

    In this context, recognised means 'known'
    in the sense of being part of some set of values
    that are understood and defined.
    For example, this error could be raised when a value
    is not be part of a set of a controlled vocabulary/known definitions.
    """

    def __init__(
        self, unrecognised_value: Any, name: Any, known_values: Collection[Any]
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        unrecognised_value
            The unrecognised value

        name
            The name of the thing that has the unrecognised value

            For example, `unrecognised_value` might be `"Emissions|junk"`
            and `name` could be `"variable"`.

        known_values
            The known values for `metadata_key`
        """
        error_msg_l = [f"{unrecognised_value!r} is not a recognised value for {name}."]
        close = difflib.get_close_matches(
            unrecognised_value, known_values, n=3, cutoff=0.6
        )
        if close:
            close_with_quotes = [f"{v!r}" for v in close]
            error_msg_l.append(f"Did you mean {' or '.join(close_with_quotes)}?")

        error_msg_l.append(f"The full list of known values is: {known_values}")

        super().__init__(" ".join(error_msg_l))
