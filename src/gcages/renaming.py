"""
Renaming between naming conventions

At present, the naming convention for both sides is implicit.
We are considering adding the concept of controlled vocabularies
to help clarify this, but have not done so yet.
"""

from __future__ import annotations

import sys
from typing import cast

import pandas as pd

import gcages.databases
from gcages.exceptions import UnrecognisedValueError

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


class SupportedNamingConventions(StrEnum):
    """Supported naming conventions"""

    GCAGES = "gcages"
    """This package's naming convention"""

    AR6_CFC_INFILLING_DB = "ar6_cfc_infilling_db"
    """
    The naming convention used in AR6's CFC infilling database

    Somehow this ended up being different to all the other conventions.
    """

    IAMC = "iamc"
    """
    Integrated Assessment Modelling Consortium (IAMC) naming convention

    Not a perfect definition so the implementation here is a bit of a guess
    based on experience.
    https://github.com/IAMconsortium/common-definitions
    is a better source of truth, but it also moves more quickly,
    is not used universally and covers many more variables
    than we care about within the gcages context.
    """

    CMIP7_SCENARIOMIP = "cmip7_scenariomip"
    """
    The naming convention used during preparation of ScenarioMIP for CMIP7
    """

    OPENSCM_RUNNER = "openscm_runner"
    """
    OpenSCM-Runner naminv convention

    Used by the package which actually runs simple climate models (SCMs),
    see https://github.com/openscm/openscm-runner
    """

    RCMIP = "rcmip"
    """
    Reduced Complexity Model Intercomparison Project (RCMIP) naming convention

    See rcmip.org and e.g.

    - https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/v5.1.0/rcmip-emissions-annual-means-v5-1-0.csv
    - https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/v5.1.0/rcmip-concentrations-annual-means-v5-1-0.csv
    - https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/v5.1.0/rcmip-radiative-forcing-annual-means-v5-1-0.csv
    """


def convert_variable_name(
    variable_in: str,
    from_convention: SupportedNamingConventions,
    to_convention: SupportedNamingConventions,
    database: pd.DataFrame = gcages.databases.EMISSIONS_VARIABLES,
) -> str:
    """
    Convert a variable name to another naming convention

    Parameters
    ----------
    variable_in
        Variable name to convert

    from_convention
        Convention to convert from

    to_convention
        Convention to convert to

    database
        Database to use for the conversions

    Returns
    -------
    :
        Converted variable name

    Raises
    ------
    UnrecognisedValueError
        `variable_in` is not a recognised value in `from_convention`
    """
    from_key = str(from_convention)
    to_key = str(to_convention)

    res_l = database.loc[database[from_key] == variable_in, to_key].tolist()

    if len(res_l) < 1:
        raise UnrecognisedValueError(
            unrecognised_value=variable_in,
            name=f"the {from_convention} naming convention",
            known_values=sorted(list(database[from_key].tolist())),
        )

    if len(res_l) > 1:  # pragma: no cover
        raise AssertionError(res_l)

    return cast(str, res_l[0])
