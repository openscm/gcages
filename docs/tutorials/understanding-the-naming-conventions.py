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
from gcages.databases import EMISSIONS_VARIABLES

# %% [markdown]
# ## Converting between naming conventions

# %%
assert (
    False
), "demo conversions and error you get if you ask for something which doesn't exist"

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
