# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Understanding the naming conventions
#
# Here we discuss the different naming conventions
# used in gcages and associated communities.
# We identify the similarities and differences
# between these conventions.
#
# As you read, keep in mind that this is just `gcages`' view.
# Others may have different interpretations of the naming conventions
# and different views of the conventions used in other communities.
# As a result, you may have to write your own, custom,
# functions to convert between naming conventions.
# Nonetheless, we hope that the conversion tools provided
# will save you at least some time by covering common use cases.

# %% [markdown]
# ## Imports

# %%
import traceback
from functools import partial

import numpy as np
import pandas as pd
import pandas_indexing  # noqa: F401
import pandas_openscm

import gcages.exceptions
from gcages.databases import EMISSIONS_VARIABLES
from gcages.renaming import SupportedNamingConventions, convert_variable_name

# %%
# Register the openscm accessor
# (pix does this on import (a side-effect pattern pandas-openscm tries to avoid),
# so there is no equivalent line)
pandas_openscm.register_pandas_accessor("openscm")

# %% [markdown]
# ## Converting between naming conventions
#
# The `convert_variable_name` function
# makes it trivial to move between naming conventions.
# The API is very simple.

# %%
convert_variable_name(
    "Emissions|CO2|Fossil",
    from_convention=SupportedNamingConventions.GCAGES,
    to_convention=SupportedNamingConventions.IAMC,
)

# %% [markdown]
# The supported naming conventions are shown below.
# At present, we have the naming conventions used in:
#
# - `gcages`
# - [OpenSCM-Runner](https://github.com/openscm/openscm-runner)
# - the IAMC community, e.g. the [IPCC AR6 database](https://data.ene.iiasa.ac.at/ar6)
# - RCMIP (rcmip.org), as used in e.g. https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/v5.1.0/rcmip-emissions-annual-means-v5-1-0.csv
# - the infilling database used in AR6 for CFCs
#   (which is somehow different from all the rest)

# %%
[v.name for v in SupportedNamingConventions]

# %% [markdown]
# Conversions between any of these naming conventions is possible

# %%
convert_variable_name(
    "Emissions|CO2|Energy and Industrial Processes",
    from_convention=SupportedNamingConventions.IAMC,
    to_convention=SupportedNamingConventions.GCAGES,
)

# %%
convert_variable_name(
    "Emissions|CO2|Energy and Industrial Processes",
    from_convention=SupportedNamingConventions.IAMC,
    to_convention=SupportedNamingConventions.OPENSCM_RUNNER,
)

# %%
convert_variable_name(
    "Emissions|CO2|Energy and Industrial Processes",
    from_convention=SupportedNamingConventions.IAMC,
    to_convention=SupportedNamingConventions.RCMIP,
)

# %%
convert_variable_name(
    "Emissions|CO2|Energy and Industrial Processes",
    from_convention=SupportedNamingConventions.IAMC,
    to_convention=SupportedNamingConventions.AR6_CFC_INFILLING_DB,
)

# %%
convert_variable_name(
    "Emissions|CO2|Fossil",
    from_convention=SupportedNamingConventions.GCAGES,
    to_convention=SupportedNamingConventions.OPENSCM_RUNNER,
)

# %%
convert_variable_name(
    "Emissions|CO2|MAGICC AFOLU",
    from_convention=SupportedNamingConventions.OPENSCM_RUNNER,
    to_convention=SupportedNamingConventions.GCAGES,
)

# %%
convert_variable_name(
    "Emissions|CO2|MAGICC AFOLU",
    from_convention=SupportedNamingConventions.OPENSCM_RUNNER,
    to_convention=SupportedNamingConventions.IAMC,
)

# %% [markdown]
# ### Errors
#
# If you try and convert a name that is not recognised,
# you will receive an `gcages.exceptions.UnrecognisedValueError`,
# which also shows close values
# and the full list of known values.

# %%
try:
    convert_variable_name(
        "Emissions|NMVOC",
        from_convention=SupportedNamingConventions.OPENSCM_RUNNER,
        to_convention=SupportedNamingConventions.GCAGES,
    )
except gcages.exceptions.UnrecognisedValueError:
    traceback.print_exc(limit=0)

# %%
try:
    convert_variable_name(
        "Emissions|junk",
        from_convention=SupportedNamingConventions.OPENSCM_RUNNER,
        to_convention=SupportedNamingConventions.GCAGES,
    )
except gcages.exceptions.UnrecognisedValueError:
    traceback.print_exc(limit=0)

# %% [markdown]
# ### Applying to pandas
#
# You can obviously apply these functions to pandas DataFrame's.
# When combined with other packages like
# [pandas-indexing](https://pandas-indexing.readthedocs.io/en/latest/)
# or [pandas-openscm](https://pandas-openscm.readthedocs.io/en/latest/),
# this can make data manipulation and conversion very straightforward.

# %%
start = pd.DataFrame(
    np.arange(12).reshape((4, 3)),
    columns=[2010, 2020, 2030],
    index=pd.MultiIndex.from_tuples(
        [
            ("sa", "Emissions|CO2|Fossil", "Mt CO2/yr"),
            ("sa", "Emissions|CO2|Biosphere", "Mt CO2/yr"),
            ("sa", "Emissions|SOx", "Mt S/yr"),
            ("sa", "Emissions|NMVOC", "Mt VOC/yr"),
        ],
        names=["scenario", "variable", "unit"],
    ),
)
start

# %%
convert_gcages_variable_to_iamc = partial(
    convert_variable_name,
    from_convention=SupportedNamingConventions.GCAGES,
    to_convention=SupportedNamingConventions.IAMC,
)
start.openscm.update_index_levels(
    {"variable": convert_gcages_variable_to_iamc, "scenario": {"sa": "scenario a"}}
)

# %%
start.pix.assign(
    variable=start.index.pix.project("variable").map(convert_gcages_variable_to_iamc),
    scenario="scenario a",
)

# %% [markdown]
# ## The 'database'
#
# `gcages` comes with a 'database'
# (in quotes because it's not really a database
# like is used in web, it's just a pandas DataFrame,
# although it serves the same purpose and has the same shape/behaviour).
# This database stores the mapping between the naming conventions
# used in different communites.

# %%
# The database in full
EMISSIONS_VARIABLES

# %% [markdown]
# The whole table above shows you the full mapping.
# However, it is helpful to break it down a bit to see where the differences are.

# %% [markdown]
# ### gcages vs. OpenSCM-Runner
#
# The differences here are quite minor, essentially clarifying
# what the CO<sub>2</sub> sub-sectors are
# (whether the CO<sub>2</sub> came from fossil or biosphere reservoirs)
# and then clearer names for emissions of sulfates (which are not pure sulfur)
# and emissions of non-methane volatile organic compounds
# (the non-methane part is dropped in some naming conventions for some reason).

# %%
disp = EMISSIONS_VARIABLES[
    EMISSIONS_VARIABLES["gcages"] != EMISSIONS_VARIABLES["openscm_runner"]
][["gcages", "openscm_runner"]]
disp

# %% [markdown]
# ### gcages vs. IAMC
#
# The differences here are more substantial,
# affecting the majority of variables.
# The IAMC convention is to include groupings within the variable name.
# This is not used by `gcages` because they generally get in the way
# and there are multiple groupings of interest, so we don't pick one in particular.
# The IAMC uses groupings like:
#
# - PFCs
# - HFCs
# - Montreal Gases
#
# There is also a difference in the naming of CO<sub>2</sub> sub-sectors,
# with the `gcages` convention again used
# for clarity of the source of the CO<sub>2</sub>,
# and clearer names for emissions of sulfates (which are not pure sulfur)
# and non-methane volatile organic compounds
# (the non-methane part is dropped in the IAMC convention).

# %%
disp = EMISSIONS_VARIABLES[
    EMISSIONS_VARIABLES["gcages"] != EMISSIONS_VARIABLES["iamc"]
][["gcages", "iamc"]]
disp

# %%
disp_same = EMISSIONS_VARIABLES[
    EMISSIONS_VARIABLES["gcages"] == EMISSIONS_VARIABLES["iamc"]
][["gcages", "iamc"]]
disp_same

# %% [markdown]
# ### gcages vs. RCMIP
#
# The differences here affect almost every variable.
# The RCMIP convention is to include groupings within the variable name.
# This is not used by `gcages` because they generally get in the way
# and there are multiple groupings of interest, so we don't pick one in particular.
# The RCMIP convention uses highly specified groupings like:
#
# - PFCs
# - HFCs
# - Montreal Gases
#
# There is also a difference in the naming of CO<sub>2</sub> sub-sectors,
# with the `gcages` convention again used
# for clarity of the source of the CO<sub>2</sub>,
# and clearer names for emissions of sulfates (which are not pure sulfur)
# and non-methane volatile organic compounds
# (the non-methane part is dropped in the IAMC convention).

# %%
disp = EMISSIONS_VARIABLES[
    EMISSIONS_VARIABLES["gcages"] != EMISSIONS_VARIABLES["rcmip"]
][["gcages", "rcmip"]]
disp

# %%
disp_same = EMISSIONS_VARIABLES[
    EMISSIONS_VARIABLES["gcages"] == EMISSIONS_VARIABLES["rcmip"]
][["gcages", "rcmip"]]
disp_same

# %% [markdown]
# ### gcages vs. AR6 CFC infilling database
#
# The differences here are for PFCs and HFCs mostly.
# The AR6 CFC infilling database convention
# is to include groupings within the variable name.
# This is not used by `gcages` because they generally get in the way
# and there are multiple groupings of interest, so we don't pick one in particular.
#
# There is also a difference in the naming of CO<sub>2</sub> sub-sectors,
# with the `gcages` convention again used
# for clarity of the source of the CO<sub>2</sub>,
# and clearer names for emissions of sulfates (which are not pure sulfur)
# and non-methane volatile organic compounds
# (the non-methane part is dropped in the IAMC convention).

# %%
disp = EMISSIONS_VARIABLES[
    EMISSIONS_VARIABLES["gcages"] != EMISSIONS_VARIABLES["ar6_cfc_infilling_db"]
][["gcages", "ar6_cfc_infilling_db"]]
disp

# %%
disp_same = EMISSIONS_VARIABLES[
    EMISSIONS_VARIABLES["gcages"] == EMISSIONS_VARIABLES["ar6_cfc_infilling_db"]
][["gcages", "ar6_cfc_infilling_db"]]
disp_same
