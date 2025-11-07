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

# %%
import pandas as pd

# Load the inventory
df = pd.read_csv("/Users/tessamoller/Documents/flat10/flat10_inventory2.csv")

df.sample(20)


# %%
# --- 1. See what unique dimension combinations exist ---
unique_dims = df["Dimensions"].value_counts()
print("Unique dimension sets:")
print(unique_dims)

# --- 2. Group by variable name to see dimension diversity ---
dims_by_var = df.groupby("Variable")["Dimensions"].unique()
print("\nDimension sets by variable:")
print(dims_by_var)

# --- 3. Check how many variables have cell_methods defined ---
cell_methods_counts = df["Cell_methods"].value_counts(dropna=False)
print("\nCell methods distribution:")
print(cell_methods_counts)


# %%
print("Unique variables:")
print(np.sort(df["Variable"].unique()))

# %%
# Find runs that have zos
runs_with_zos = df.loc[df["Variable"] == "zos", "Path"].unique()

# Find runs that have zostoga
runs_with_zostoga = df.loc[df["Variable"] == "zostoga", "Path"].unique()

print("Models with zos:", runs_with_zos)
print("Models with zostoga:", runs_with_zostoga)
