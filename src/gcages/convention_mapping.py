"""
Tools from mapping from one set of conventions to another

A lot of these should be given a new home
or translated into upstream fixes.
"""

from __future__ import annotations

from collections.abc import Iterable

from gcages.exceptions import MissingOptionalDependencyError


def transform_iamc_to_openscm_runner_variable(v):
    """
    Transform IAMC variable to OpenSCM-Runner variable

    Parameters
    ----------
    v
        Variable name to transform

    Returns
    -------
    :
        OpenSCM-Runner equivalent of `v`
    """
    res = v

    replacements = (
        ("CFC|", ""),
        ("HFC|", ""),
        ("PFC|", ""),
        ("|Montreal Gases", ""),
        (
            "HFC43-10",
            "HFC4310mee",
        ),
        (
            "AFOLU",
            "MAGICC AFOLU",
        ),
        (
            "Energy and Industrial Processes",
            "MAGICC Fossil and Industrial",
        ),
    )
    for old, new in replacements:
        res = res.replace(old, new)

    return res


def convert_openscm_runner_output_names_to_magicc_output_names(
    openscm_runner_names: Iterable[str],
) -> tuple[str, ...]:
    """
    Get output names for the call to MAGICC

    Parameters
    ----------
    openscm_runner_names
        OpenSCM-Runner output names

    Returns
    -------
    :
        MAGICC output names
    """
    try:
        import pymagicc.definitions
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "convert_openscm_runner_output_names_to_magicc_output_names",
            requirement="pymagicc",
        ) from exc

    res_l = []
    for openscm_runner_variable in openscm_runner_names:
        if openscm_runner_variable == "Surface Air Temperature Change":
            # A fun bug in pymagicc
            res_l.append("SURFACE_TEMP")
        elif openscm_runner_variable == "Effective Radiative Forcing|HFC4310mee":
            # Another fun bug in pymagicc
            magicc_var = pymagicc.definitions.convert_magicc7_to_openscm_variables(
                "Effective Radiative Forcing|HFC4310",
                inverse=True,
            )
            res_l.append(magicc_var)
        else:
            magicc_var = pymagicc.definitions.convert_magicc7_to_openscm_variables(
                openscm_runner_variable,
                inverse=True,
            )
            res_l.append(magicc_var)

    return tuple(res_l)
