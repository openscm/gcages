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

import numpy as np
import pandas as pd
import pandas_indexing  # noqa: F401
import pandas_openscm

import gcages.exceptions
from gcages.databases import EMISSIONS_VARIABLES
from gcages.renaming import (
    convert_gcages_variable_to_iamc,
    convert_gcages_variable_to_openscm_runner,
    convert_iamc_variable_to_gcages,
    convert_iamc_variable_to_openscm_runner,
    convert_openscm_runner_variable_to_gcages,
    convert_openscm_runner_variable_to_iamc,
)

# %%
# Register the openscm accessor
# (pix does this on import (a side-effect pattern pandas-openscm tries to avoid),
# so there is no equivalent line)
pandas_openscm.register_pandas_accessor("openscm")

# %% [markdown]
# ## Converting between naming conventions
#
# The renaming functions make it trivial to move between naming conventions.
# Their API is very simple.

# %%
convert_gcages_variable_to_iamc("Emissions|CO2|Fossil")

# %%
convert_gcages_variable_to_openscm_runner("Emissions|CO2|Fossil")

# %%
convert_iamc_variable_to_gcages("Emissions|CO2|Energy and Industrial Processes")

# %%
convert_iamc_variable_to_openscm_runner("Emissions|CO2|Energy and Industrial Processes")

# %%
convert_openscm_runner_variable_to_gcages("Emissions|CO2|MAGICC AFOLU")

# %%
convert_openscm_runner_variable_to_iamc("Emissions|CO2|MAGICC AFOLU")

# %% [markdown]
# ### Errors
#
# If you try and convert a name that is not recognised,
# you will receive an `gcages.exceptions.UnrecognisedValueError`,
# which also shows close values
# and the full list of known values.

# %%
try:
    convert_openscm_runner_variable_to_gcages("Emissions|NMVOC")
except gcages.exceptions.UnrecognisedValueError:
    traceback.print_exc(limit=0)

# %%
try:
    convert_openscm_runner_variable_to_gcages("Emissions|junk")
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
# At present, we have the naming conventions used in:
#
# - `gcages`
# - [OpenSCM-Runner](https://github.com/openscm/openscm-runner)
# - the IAMC community, e.g. the [IPCC AR6 database](https://data.ene.iiasa.ac.at/ar6)

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
# ### OpenSCM-Runner vs. IAMC
#
# The differences here are basically the same as gcages compared to IAMC.
# As above, the IAMC convention is to include groupings within the variable name,
# which leads to differences.
# There is, as above, a difference in the naming of CO<sub>2</sub> sub-sectors,
# although there is no difference in naming
# for emissions of sulfates or non-methane volatile organic compounds
# between these two naming conventions.

# %%
disp = EMISSIONS_VARIABLES[
    EMISSIONS_VARIABLES["openscm_runner"] != EMISSIONS_VARIABLES["iamc"]
][["openscm_runner", "iamc"]]
disp
