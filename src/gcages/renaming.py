"""
Renaming between naming conventions

At present, the naming convention for both sides is implicit.
We are considering adding the concept of controlled vocabularies
to help clarify this, but have not done so yet.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from gcages.exceptions import UnrecognisedValueError


def lookup_mapping(
    from_key: str,
    from_value: str,
    to_key: str,
    database: pd.DataFrame,
    variable_used_for_lookup: str,
) -> str:
    """
    Lookup a mapping

    Parameters
    ----------
    from_key
        Key/column to map from

    from_value
        Value to map from (i.e. what to look up in the `from_key` column)

    to_key
        Key/column to map to

    database
        Database in which to look up the mapping

        (Not a real database, just a [pd.DataFrame][pandas.DataFrame],
        but it performs the same function.)

    variable_used_for_lookup
        The name of the variable being used for mapping

        This is only used to provide a helpful error message.

    Returns
    -------
    :
        Mapped value i.e. the equivalent value to `from_value` in `to_key`

    Raises
    ------
    UnrecognisedValueError
        `from_value` is not a recognised value in `from_key` for the given `database`
    """
    res_l = database.loc[database[from_key] == from_value, to_key].tolist()

    if len(res_l) < 1:
        raise UnrecognisedValueError(
            unrecognised_value=from_value,
            name=variable_used_for_lookup,
            known_values=sorted(list(database[from_key].tolist())),
        )

    if len(res_l) > 1:  # pragma: no cover
        raise AssertionError(res_l)

    return cast(str, res_l[0])


def convert_gcages_variable_to_iamc(gcages_variable: str) -> str:
    """
    Convert a gcages variable name to an IAMC variable name

    Parameters
    ----------
    gcages_variable
        gcages variable to convert

    Returns
    -------
    :
        IAMC equivalent of `gcages_variable`

    Raises
    ------
    UnrecognisedValueError
        We do not know how to map `gcages_variable`
    """
    from gcages.databases import EMISSIONS_VARIABLES

    return lookup_mapping(
        from_key="gcages",
        from_value=gcages_variable,
        variable_used_for_lookup="gcages_variable",
        to_key="iamc",
        database=EMISSIONS_VARIABLES,
    )


def convert_gcages_variable_to_openscm_runner(gcages_variable: str) -> str:
    """
    Convert a gcages variable name to an OpenSCM-Runner variable name

    Parameters
    ----------
    gcages_variable
        gcages variable to convert

    Returns
    -------
    :
        OpenSCM-Runner equivalent of `gcages_variable`
    """
    from gcages.databases import EMISSIONS_VARIABLES

    return lookup_mapping(
        from_key="gcages",
        from_value=gcages_variable,
        variable_used_for_lookup="gcages_variable",
        to_key="openscm_runner",
        database=EMISSIONS_VARIABLES,
    )


def convert_gcages_variable_to_rcmip(gcages_variable: str) -> str:
    """
    Convert a gcages variable name to an RCMIP variable name

    Parameters
    ----------
    gcages_variable
        gcages variable to convert

    Returns
    -------
    :
        RCMIP equivalent of `gcages_variable`
    """
    from gcages.databases import EMISSIONS_VARIABLES

    return lookup_mapping(
        from_key="gcages",
        from_value=gcages_variable,
        variable_used_for_lookup="gcages_variable",
        to_key="rcmip",
        database=EMISSIONS_VARIABLES,
    )


def convert_iamc_variable_to_gcages(iamc_variable: str) -> str:
    """
    Convert an IAMC variable name to a gcages variable name

    Parameters
    ----------
    iamc_variable
        IAMC variable to convert

    Returns
    -------
    :
        gcages equivalent of `iamc_variable`

    Raises
    ------
    UnrecognisedValueError
        We do not know how to map `iamc_variable`
    """
    from gcages.databases import EMISSIONS_VARIABLES

    return lookup_mapping(
        from_key="iamc",
        from_value=iamc_variable,
        variable_used_for_lookup="iamc_variable",
        to_key="gcages",
        database=EMISSIONS_VARIABLES,
    )


def convert_iamc_variable_to_openscm_runner(iamc_variable: str) -> str:
    """
    Convert an IAMC variable name to an OpenSCM-Runner variable name

    Parameters
    ----------
    iamc_variable
        IAMC variable to convert

    Returns
    -------
    :
        OpenSCM-Runner equivalent of `iamc_variable`

    Raises
    ------
    UnrecognisedValueError
        We do not know how to map `iamc_variable`
    """
    from gcages.databases import EMISSIONS_VARIABLES

    return lookup_mapping(
        from_key="iamc",
        from_value=iamc_variable,
        variable_used_for_lookup="iamc_variable",
        to_key="openscm_runner",
        database=EMISSIONS_VARIABLES,
    )


def convert_iamc_variable_to_rcmip(iamc_variable: str) -> str:
    """
    Convert an IAMC variable name to an RCMIP variable name

    Parameters
    ----------
    iamc_variable
        IAMC variable to convert

    Returns
    -------
    :
        RCMIP equivalent of `iamc_variable`

    Raises
    ------
    UnrecognisedValueError
        We do not know how to map `iamc_variable`
    """
    from gcages.databases import EMISSIONS_VARIABLES

    return lookup_mapping(
        from_key="iamc",
        from_value=iamc_variable,
        variable_used_for_lookup="iamc_variable",
        to_key="rcmip",
        database=EMISSIONS_VARIABLES,
    )


def convert_openscm_runner_variable_to_gcages(openscm_runner_variable: str) -> str:
    """
    Convert an OpenSCM-Runner variable name to a gcages variable name

    Parameters
    ----------
    openscm_runner_variable
        OpenSCM-Runner variable to convert

    Returns
    -------
    :
        gcages equivalent of `openscm_runner_variable`
    """
    from gcages.databases import EMISSIONS_VARIABLES

    return lookup_mapping(
        from_key="openscm_runner",
        from_value=openscm_runner_variable,
        variable_used_for_lookup="openscm_runner_variable",
        to_key="gcages",
        database=EMISSIONS_VARIABLES,
    )


def convert_openscm_runner_variable_to_iamc(openscm_runner_variable: str) -> str:
    """
    Convert an OpenSCM-Runner variable name to an IAMC variable name

    Parameters
    ----------
    openscm_runner_variable
        OpenSCM-Runner variable to convert

    Returns
    -------
    :
        IAMC equivalent of `openscm_runner_variable`
    """
    from gcages.databases import EMISSIONS_VARIABLES

    return lookup_mapping(
        from_key="openscm_runner",
        from_value=openscm_runner_variable,
        variable_used_for_lookup="openscm_runner_variable",
        to_key="iamc",
        database=EMISSIONS_VARIABLES,
    )


def convert_openscm_runner_variable_to_rcmip(openscm_runner_variable: str) -> str:
    """
    Convert an OpenSCM-Runner variable name to an RCMIP variable name

    Parameters
    ----------
    openscm_runner_variable
        OpenSCM-Runner variable to convert

    Returns
    -------
    :
        RCMIP equivalent of `openscm_runner_variable`
    """
    from gcages.databases import EMISSIONS_VARIABLES

    return lookup_mapping(
        from_key="openscm_runner",
        from_value=openscm_runner_variable,
        variable_used_for_lookup="openscm_runner_variable",
        to_key="rcmip",
        database=EMISSIONS_VARIABLES,
    )


def convert_rcmip_variable_to_gcages(rcmip_variable: str) -> str:
    """
    Convert an RCMIP variable name to a gcages variable name

    Parameters
    ----------
    rcmip_variable
        RCMIP variable to convert

    Returns
    -------
    :
        gcages equivalent of `rcmip_variable`

    Raises
    ------
    UnrecognisedValueError
        We do not know how to map `rcmip_variable`
    """
    from gcages.databases import EMISSIONS_VARIABLES

    return lookup_mapping(
        from_key="rcmip",
        from_value=rcmip_variable,
        variable_used_for_lookup="rcmip_variable",
        to_key="gcages",
        database=EMISSIONS_VARIABLES,
    )


def convert_rcmip_variable_to_iamc(rcmip_variable: str) -> str:
    """
    Convert an RCMIP variable name to an IAMC variable name

    Parameters
    ----------
    rcmip_variable
        RCMIP variable to convert

    Returns
    -------
    :
        RCMIP equivalent of `rcmip_variable`

    Raises
    ------
    UnrecognisedValueError
        We do not know how to map `rcmip_variable`
    """
    from gcages.databases import EMISSIONS_VARIABLES

    return lookup_mapping(
        from_key="rcmip",
        from_value=rcmip_variable,
        variable_used_for_lookup="rcmip_variable",
        to_key="iamc",
        database=EMISSIONS_VARIABLES,
    )


def convert_rcmip_variable_to_openscm_runner(rcmip_variable: str) -> str:
    """
    Convert an RCMIP variable name to an OpenSCM-Runner variable name

    Parameters
    ----------
    rcmip_variable
        RCMIP variable to convert

    Returns
    -------
    :
        OpenSCM-Runner equivalent of `rcmip_variable`

    Raises
    ------
    UnrecognisedValueError
        We do not know how to map `rcmip_variable`
    """
    from gcages.databases import EMISSIONS_VARIABLES

    return lookup_mapping(
        from_key="rcmip",
        from_value=rcmip_variable,
        variable_used_for_lookup="rcmip_variable",
        to_key="openscm_runner",
        database=EMISSIONS_VARIABLES,
    )
