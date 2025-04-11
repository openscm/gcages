"""
Renaming between naming conventions

At present, the naming convention for both sides is implicit.
We are considering adding the concept of controlled vocabularies
to help clarify this, but have not done so yet.
"""

from __future__ import annotations


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
    if "HFC" in iamc_variable:
        toks = iamc_variable.split(separator)
        res = separator.join([toks[0], toks[-1]])

        if res.endswith(f"{separator}HFC43-10"):
            res = res.replace(f"{separator}HFC43-10", f"{separator}HFC4310mee")

        return res

    if "Montreal Gases" in iamc_variable:
        toks = iamc_variable.split(separator)
        res = separator.join([toks[0], toks[-1]])

        return res

    if iamc_variable.endswith(f"{separator}Sulfur"):
        return iamc_variable.replace(f"{separator}Sulfur", f"{separator}SOx")

    if iamc_variable.endswith(f"{separator}VOC"):
        return iamc_variable.replace(f"{separator}VOC", f"{separator}NMVOC")

    return iamc_variable


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
        gcages equivalent of `iamc_variable`
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
