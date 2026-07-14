"""
Emissions variables database
"""

from __future__ import annotations

import importlib.resources

import pandas as pd

EMISSIONS_VARIABLES = pd.read_csv(
    importlib.resources.files("gcages.databases").joinpath("emissions_variables.csv")  # type: ignore # some pathlib bug (result should just be a path...)
)
"""
Database of emissions variables names according to different naming schemes

You will likely not need to access this variable directly,
and instead will use [convert_variable_name][gcages.renaming.].
"""
