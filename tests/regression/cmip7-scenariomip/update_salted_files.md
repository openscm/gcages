# Update Salted Files

- Update Date: 08 Jul 2025
- Author: Marco Zecchetto

Updated (i.e. overwritten) files `salted-202504-scenariomip-input_global_workflow_emissions_raw_names.csv`, `salted-202504-scenariomip-input_global_workflow_emissions.csv`, `salted-202504-scenariomip-input_gridding_workflow_emissions.csv`.

Added a breakpoint in line 49 of `test_regression_cmip7_scenariomip_pre_processing.py` and saved to csv the res Dataframe of interest. e.g. `res.global_workflow_emissions.to_csv("salted-202504-scenariomip-input_global_workflow_emissions.csv")`.

(END)
