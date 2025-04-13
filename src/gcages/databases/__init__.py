"""
Databases used by the package

They're not really databases for the most part
(rather just [pd.DataFrame][pandas.DataFrame]'s),
but the concept is the same.
"""

from __future__ import annotations

from gcages.databases.emissions_variables import EMISSIONS_VARIABLES

__all__ = ["EMISSIONS_VARIABLES"]
