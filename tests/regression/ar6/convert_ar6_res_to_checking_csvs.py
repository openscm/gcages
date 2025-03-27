"""
Convert the test file from xls to csvs we can use in the repo
"""

from pathlib import Path

HERE = Path(__file__).parents[0]
AR6_OUTPUT_DIR = HERE / "ar6-output"
OUTPUT_DIR = HERE / "ar6-output-processed"
# This file is available from the scenario explorer.
# It's nearly 1GB, which is why we don't include it in the repository.
# It's md5sum is:
# 4b8aea8270566584db9d893ff7b4562e  ar6-output/AR6_Scenarios_Database_World_ALL_CLIMATE_v1.1.csv  # noqa: E501
# (Generated with `md5sum ar6-output/AR6_Scenarios_Database_World_ALL_CLIMATE_v1.1.csv`)  # noqa: E501


def main():
    import pandas as pd
    import pandas_indexing as pix
    import tqdm

    raw = pd.read_csv(
        AR6_OUTPUT_DIR / "AR6_Scenarios_Database_World_ALL_CLIMATE_v1.1.csv"
    ).set_index(["Model", "Scenario", "Variable", "Unit", "Region"])

    emissions = raw[pix.ismatch(Variable="**Emissions**")]
    temperature_magicc = raw[pix.ismatch(Variable="**|Surface Temperature**MAGICC**")]

    raw_index = temperature_magicc.index
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    raw_index.to_frame()[["Model", "Scenario"]].drop_duplicates().to_csv(
        OUTPUT_DIR / "ar6_scenarios_raw_model_scenario_combinations.csv", index=False
    )

    for (model, scenario), msdf in tqdm.tqdm(
        temperature_magicc.groupby(["Model", "Scenario"])
    ):
        filename = f"ar6_scenarios__{model}__{scenario}__temperatures.csv"
        filename = filename.replace("/", "_").replace(" ", "_")
        out_file = OUTPUT_DIR / filename

        msdf.to_csv(out_file)

        filename_emissions = f"ar6_scenarios__{model}__{scenario}__emissions.csv"
        filename_emissions = filename_emissions.replace("/", "_").replace(" ", "_")
        out_file_emissions = OUTPUT_DIR / filename_emissions
        emissions[pix.ismatch(Model=model) & pix.ismatch(Scenario=scenario)].to_csv(
            out_file_emissions
        )


if __name__ == "__main__":
    main()
