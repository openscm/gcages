"""
Extract unique testing scenarios from a scenario set

This removes any scenarios which have the same reporting
in terms of variables and times.
Such duplicates are the same from our processing's point of view.
There is no point testing more than one of them.
"""

from pathlib import Path

import pandas as pd
import pandas_indexing as pix


def main() -> None:
    """
    Extract the unique testing scenarios
    """
    infile = Path(
        "scripts/SCI-2025_v1.1_beta.2_pathways_ensemble_global_emissions.xlsx"
    )
    start = pd.read_excel(infile, sheet_name="data")
    # infile = Path("SCI-2025_v1.1_beta_pathways_ensemble_global_emissions.feather")
    # start = pd.read_feather(infile)

    outfile = Path("SCI-2026-June-unique-testing-pathways.csv")

    # This defines what our 'starting set to consider is'.
    #
    # For SCI, the maximum set of emissions we can consider is the set below.
    # Any variations beyond these, we don't care about.
    starting_filter = (
        pix.ismatch(region="World")
        & pix.ismatch(
            variable=[
                "Emissions|*",
                "Emissions|HFC|**",
                # What we consider when trying to figure out the CO2 tree
                "Emissions|CO2",
                "Emissions|CO2|Energy and Industrial Processes",
                "Emissions|CO2|AFOLU",
                "Emissions|CO2|Other",
                "Emissions|CO2|Waste",
                "Emissions|CO2|Other Capture and Removal",
                "Emissions|CO2|Product Use",
            ]
        )
        & ~pix.isin(
            variable=[
                "Emissions|F-Gases",
                "Emissions|PFC",
                "Emissions|HFC",
                "Emissions|Kyoto Gases",
            ]
        )
    )
    starting_time_filter = slice(2020, None)

    start_clean = start.copy()
    start_clean.columns = start_clean.columns.str.lower()
    start_clean = start_clean.set_index(
        ["model", "scenario", "region", "variable", "unit"]
    )
    start_clean.columns = start_clean.columns.astype(int)
    start_clean = start_clean.sort_index(axis="columns")

    start_filtered = start_clean.loc[starting_filter, starting_time_filter]

    unique_inputs_l = []
    input_maps_included = []
    for (model, scenario), msdf in start_filtered.groupby(["model", "scenario"]):
        msdf_null = msdf.isnull().reset_index(["model", "scenario"], drop=True)

        already_included = False
        for v in input_maps_included:
            if (
                v.columns.equals(msdf_null.columns)
                and v.index.equals(msdf_null.index)
                and (msdf_null == v).all(axis=None)
            ):
                already_included = True
                break

        if not already_included:
            unique_inputs_l.append(msdf)
            input_maps_included.append(msdf_null)

    n_unique_scenarios_start = (
        start_filtered.index.to_frame()[["model", "scenario"]]
        .drop_duplicates()
        .shape[0]
    )
    msg = (
        f"Started with {n_unique_scenarios_start} scenarios. "
        f"Extracted {len(unique_inputs_l)} unique variations of interest."
    )
    print(msg)
    unique_inputs = pix.concat(unique_inputs_l)
    unique_inputs.to_csv(outfile)


if __name__ == "__main__":
    main()
