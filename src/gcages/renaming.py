"""
Renaming between naming conventions

At present, the naming convention for both sides is implicit.
We are considering adding the concept of controlled vocabularies
to help clarify this, but have not done so yet.
"""

from __future__ import annotations

import functools

from gcages.exceptions import UnrecognisedValueError

IAMC_TO_GCAGES_TOKS_MAP: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("Emissions", "BC"), ("Emissions", "BC")),
    (("Emissions", "C2F6"), ("Emissions", "C2F6")),
    (("Emissions", "C3F8"), ("Emissions", "C3F8")),
    (("Emissions", "C4F10"), ("Emissions", "C4F10")),
    (("Emissions", "C5F12"), ("Emissions", "C5F12")),
    (("Emissions", "C6F14"), ("Emissions", "C6F14")),
    (("Emissions", "C7F16"), ("Emissions", "C7F16")),
    (("Emissions", "C8F18"), ("Emissions", "C8F18")),
    (("Emissions", "CF4"), ("Emissions", "CF4")),
    (("Emissions", "CH4"), ("Emissions", "CH4")),
    (("Emissions", "CO"), ("Emissions", "CO")),
    (("Emissions", "CO2", "AFOLU"), ("Emissions", "CO2", "Biosphere")),
    (
        ("Emissions", "CO2", "Energy and Industrial Processes"),
        ("Emissions", "CO2", "Fossil"),
    ),
    (("Emissions", "HFC", "HFC125"), ("Emissions", "HFC125")),
    (("Emissions", "HFC", "HFC134a"), ("Emissions", "HFC134a")),
    (("Emissions", "HFC", "HFC143a"), ("Emissions", "HFC143a")),
    (("Emissions", "HFC", "HFC152a"), ("Emissions", "HFC152a")),
    (("Emissions", "HFC", "HFC227ea"), ("Emissions", "HFC227ea")),
    (("Emissions", "HFC", "HFC23"), ("Emissions", "HFC23")),
    (("Emissions", "HFC", "HFC236fa"), ("Emissions", "HFC236fa")),
    (("Emissions", "HFC", "HFC245fa"), ("Emissions", "HFC245fa")),
    (("Emissions", "HFC", "HFC32"), ("Emissions", "HFC32")),
    (("Emissions", "HFC", "HFC365mfc"), ("Emissions", "HFC365mfc")),
    (("Emissions", "HFC", "HFC43-10"), ("Emissions", "HFC4310mee")),
    (("Emissions", "Montreal Gases", "CCl4"), ("Emissions", "CCl4")),
    (("Emissions", "Montreal Gases", "CFC", "CFC11"), ("Emissions", "CFC11")),
    (("Emissions", "Montreal Gases", "CFC", "CFC113"), ("Emissions", "CFC113")),
    (("Emissions", "Montreal Gases", "CFC", "CFC114"), ("Emissions", "CFC114")),
    (("Emissions", "Montreal Gases", "CFC", "CFC115"), ("Emissions", "CFC115")),
    (("Emissions", "Montreal Gases", "CFC", "CFC12"), ("Emissions", "CFC12")),
    (("Emissions", "Montreal Gases", "CH2Cl2"), ("Emissions", "CH2Cl2")),
    (("Emissions", "Montreal Gases", "CH3Br"), ("Emissions", "CH3Br")),
    (("Emissions", "Montreal Gases", "CH3CCl3"), ("Emissions", "CH3CCl3")),
    (("Emissions", "Montreal Gases", "CH3Cl"), ("Emissions", "CH3Cl")),
    (("Emissions", "Montreal Gases", "CHCl3"), ("Emissions", "CHCl3")),
    (("Emissions", "Montreal Gases", "HCFC141b"), ("Emissions", "HCFC141b")),
    (("Emissions", "Montreal Gases", "HCFC142b"), ("Emissions", "HCFC142b")),
    (("Emissions", "Montreal Gases", "HCFC22"), ("Emissions", "HCFC22")),
    (("Emissions", "Montreal Gases", "Halon1202"), ("Emissions", "Halon1202")),
    (("Emissions", "Montreal Gases", "Halon1211"), ("Emissions", "Halon1211")),
    (("Emissions", "Montreal Gases", "Halon1301"), ("Emissions", "Halon1301")),
    (("Emissions", "Montreal Gases", "Halon2402"), ("Emissions", "Halon2402")),
    (("Emissions", "N2O"), ("Emissions", "N2O")),
    (("Emissions", "NF3"), ("Emissions", "NF3")),
    (("Emissions", "NH3"), ("Emissions", "NH3")),
    (("Emissions", "NOx"), ("Emissions", "NOx")),
    (("Emissions", "OC"), ("Emissions", "OC")),
    (("Emissions", "SF6"), ("Emissions", "SF6")),
    (("Emissions", "SO2F2"), ("Emissions", "SO2F2")),
    (("Emissions", "Sulfur"), ("Emissions", "SOx")),
    (("Emissions", "VOC"), ("Emissions", "NMVOC")),
    (("Emissions", "cC4F8"), ("Emissions", "cC4F8")),
)
"""
Mapping from IAMC components (tokens) to gcages components

We keep these as tokens so that the separator can be injected as needed.
You will likely not need to access this map directly,
and instead will use [get_iamc_to_gcages_map][(m).].
"""


@functools.cache
def get_iamc_to_gcages_map(separator: str) -> dict[str, str]:
    """
    Get a map from IAMC variables to gcages variables for a given separator

    Parameters
    ----------
    separator
        Separator to use between variable components

    Returns
    -------
    :
        Map from IAMC variables to gcages variables
    """
    res = {
        separator.join(iamc_toks): separator.join(gcages_toks)
        for iamc_toks, gcages_toks in IAMC_TO_GCAGES_TOKS_MAP
    }

    return res


def convert_iamc_variable_to_gcages(iamc_variable: str, separator: str = "|") -> str:
    """
    Convert an IAMC variable name to a gcages variable name

    Parameters
    ----------
    iamc_variable
        IAMC variable to convert

    separator
        Separator between levels within the IAMC variable

    Returns
    -------
    :
        gcages equivalent of `iamc_variable`
    """
    mapping = get_iamc_to_gcages_map(separator)

    try:
        return mapping[iamc_variable]
    except KeyError as exc:
        raise UnrecognisedValueError(
            unrecognised_value=iamc_variable,
            metadata_key="iamc_variable",
            known_values=list(mapping.keys()),
        ) from exc
    # if "HFC" in iamc_variable:
    #     toks = iamc_variable.split(separator)
    #     res = separator.join([toks[0], toks[-1]])
    #
    #     if res.endswith(f"{separator}HFC43-10"):
    #         res = res.replace(f"{separator}HFC43-10", f"{separator}HFC4310mee")
    #
    #     return res
    #
    # if "Montreal Gases" in iamc_variable:
    #     toks = iamc_variable.split(separator)
    #     res = separator.join([toks[0], toks[-1]])
    #
    #     return res
    #
    # if iamc_variable.endswith(f"{separator}Sulfur"):
    #     return iamc_variable.replace(f"{separator}Sulfur", f"{separator}SOx")
    #
    # if iamc_variable.endswith(f"{separator}VOC"):
    #     return iamc_variable.replace(f"{separator}VOC", f"{separator}NMVOC")
    #
    # if iamc_variable.endswith(f"{separator}CO2{separator}AFOLU"):
    #     return iamc_variable.replace(f"{separator}AFOLU", f"{separator}Biosphere")
    #
    # if iamc_variable.endswith(
    #     f"{separator}CO2{separator}Energy and Industrial Processes"
    # ):
    #     return iamc_variable.replace(
    #         f"{separator}Energy and Industrial Processes", f"{separator}Fossil"
    #     )
    #
    # return iamc_variable


def convert_gcages_variable_to_iamc(gcages_variable: str, separator: str = "|") -> str:
    """
    Convert a gcages variable name to an IAMC variable name

    Parameters
    ----------
    gcages_variable
        gcages variable to convert

    separator
        Separator between levels within the gcages variable

    Returns
    -------
    :
        IAMC equivalent of `gcages_variable`
    """
    if "HFC" in gcages_variable:
        toks = gcages_variable.split(separator)

        res = separator.join([toks[0], "HFC", *toks[1:]])
        if res.endswith(f"{separator}HFC4310mee"):
            res = res.replace(f"{separator}HFC4310mee", f"{separator}HFC43-10")

        return res

    montreal_gases = (
        "CCl4",
        "CFC11",
        "CFC113",
        "CFC114",
        "CFC115",
        "CFC12",
        "CH2Cl2",
        "CH3Br",
        "CH3CCl3",
        "CH3Cl",
        "CHCl3",
        "HCFC141b",
        "HCFC142b",
        "HCFC22",
        "Halon1202",
        "Halon1211",
        "Halon1301",
        "Halon2402",
    )
    if any(gcages_variable.endswith(suffix) for suffix in montreal_gases):
        toks = gcages_variable.split(separator)

        insertions = ["Montreal Gases"]
        if f"{separator}CFC" in gcages_variable:
            insertions.append("CFC")

        res = separator.join([toks[0], *insertions, *toks[1:]])

        return res

    if gcages_variable.endswith(f"{separator}SOx"):
        return gcages_variable.replace(f"{separator}SOx", f"{separator}Sulfur")

    if gcages_variable.endswith(f"{separator}NMVOC"):
        return gcages_variable.replace(f"{separator}NMVOC", f"{separator}VOC")

    return gcages_variable


def convert_openscm_runner_variable_to_gcages(
    openscm_runner_variable: str, separator: str = "|"
) -> str:
    """
    Convert an OpenSCM-Runner variable name to a gcages variable name

    Parameters
    ----------
    openscm_runner_variable
        OpenSCM-Runner variable to convert

    separator
        Separator between levels within the openscm_runner variable

    Returns
    -------
    :
        gcages equivalent of `openscm_runner_variable`
    """
    if openscm_runner_variable.endswith(f"{separator}Sulfur"):
        return openscm_runner_variable.replace(f"{separator}Sulfur", f"{separator}SOx")

    if openscm_runner_variable.endswith(f"{separator}VOC"):
        return openscm_runner_variable.replace(f"{separator}VOC", f"{separator}NMVOC")

    if openscm_runner_variable.endswith(f"{separator}CO2{separator}AFOLU"):
        return openscm_runner_variable.replace(
            f"{separator}AFOLU", f"{separator}Biosphere"
        )

    return openscm_runner_variable


def convert_gcages_variable_to_openscm_runner(
    gcages_variable: str, separator: str = "|"
) -> str:
    """
    Convert a gcages variable name to an OpenSCM-Runner variable name

    Parameters
    ----------
    gcages_variable
        gcages variable to convert

    separator
        Separator between levels within the gcages variable

    Returns
    -------
    :
        OpenSCM-Runner equivalent of `gcages_variable`
    """
    if "HFC" in gcages_variable:
        toks = gcages_variable.split(separator)

        res = separator.join([toks[0], "HFC", *toks[1:]])
        if res.endswith(f"{separator}HFC4310mee"):
            res = res.replace(f"{separator}HFC4310mee", f"{separator}HFC43-10")

        return res

    montreal_gases = (
        "CCl4",
        "CFC11",
        "CFC113",
        "CFC114",
        "CFC115",
        "CFC12",
        "CH2Cl2",
        "CH3Br",
        "CH3CCl3",
        "CH3Cl",
        "CHCl3",
        "HCFC141b",
        "HCFC142b",
        "HCFC22",
        "Halon1202",
        "Halon1211",
        "Halon1301",
        "Halon2402",
    )
    if any(gcages_variable.endswith(suffix) for suffix in montreal_gases):
        toks = gcages_variable.split(separator)

        insertions = ["Montreal Gases"]
        if f"{separator}CFC" in gcages_variable:
            insertions.append("CFC")

        res = separator.join([toks[0], *insertions, *toks[1:]])

        return res

    if gcages_variable.endswith(f"{separator}SOx"):
        return gcages_variable.replace(f"{separator}SOx", f"{separator}Sulfur")

    if gcages_variable.endswith(f"{separator}NMVOC"):
        return gcages_variable.replace(f"{separator}NMVOC", f"{separator}VOC")

    return gcages_variable
