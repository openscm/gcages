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
# However, it is helpful to break it down a bit to see where
