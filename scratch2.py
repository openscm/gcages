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
from pathlib import Path
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB
import pandas_indexing as pix

scm_output_db = OpenSCMDB(
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
    db_dir=Path("scm-output-db"),
)

# %%
idx = scm_output_db.load_index()

# %%
# TODO: add partial delete to openscmdb

# %%
idx.loc[pix.isin(variable="Surface Air Temperature Change")].groupby(["model", "scenario"]).count()["file_id"].sort_values()

# %%
