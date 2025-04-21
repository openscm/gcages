"""
Constants used throughout the pre-processing

Throughout this module, we aim to follow the following conventions:

- "World" means global total
- "Required" means that the thing of interest is compulsory in the input data
- "INPUT" means that the values are in the input data naming convention
"""

from __future__ import annotations

INTERNATIONAL_AVIATION_SECTOR_INPUT: str = (
    "Energy|Demand|Bunkers|International Aviation"
)
"""
Name for the international aviation sector (input naming convention)
"""

DOMESTIC_AVIATION_SECTOR_INPUT: str = "Energy|Demand|Transportation|Domestic Aviation"
"""
Name for the domestic aviation sector (input naming convention)
"""

TRANSPORTATION_SECTOR_INPUT: str = "Energy|Demand|Transportation"
"""
Name for the transportation sector as a whole (input naming convention)
"""

REQUIRED_GRIDDING_SPECIES_INPUT: tuple[str, ...] = (
    "CO2",
    "CH4",
    "N2O",
    "BC",
    "CO",
    "NH3",
    "OC",
    "NOx",
    "Sulfur",
    "VOC",
)
"""
Required species for gridding (in the input naming convention)
"""

REQUIRED_GRIDDING_SECTORS_WORLD_INPUT: tuple[str, ...] = (
    INTERNATIONAL_AVIATION_SECTOR_INPUT,
    "Energy|Demand|Bunkers|International Shipping",
)
"""
Sectors (input naming) required in the input at the world level for gridding
"""

REQUIRED_WORLD_VARIABLES_INPUT: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_INPUT
    for sector in REQUIRED_GRIDDING_SECTORS_WORLD_INPUT
)
"""
Variables required at the world level (input naming convention)
"""

OPTIONAL_GRIDDING_SECTORS_WORLD_INPUT: tuple[str, ...] = ()
"""
Sectors (input naming) that are optional at the world level for gridding
"""

OPTIONAL_WORLD_VARIABLES_INPUT: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_INPUT
    for sector in OPTIONAL_GRIDDING_SECTORS_WORLD_INPUT
)
"""
Optional variables at the world level (input naming convention)
"""

ALL_WORLD_VARIABLES_INPUT: tuple[str, ...] = tuple(
    (*REQUIRED_WORLD_VARIABLES_INPUT, *OPTIONAL_WORLD_VARIABLES_INPUT)
)
"""
All variables considered at the world level (input naming convention)
"""

INDEPENDENT_WORLD_VARIABLES_INPUT: tuple[str, ...] = ALL_WORLD_VARIABLES_INPUT
"""
Variables that are independent at the world level (input naming convention)
"""

INDUSTRIAL_SECTOR_GRIDDING_COMPONENTS_INPUT: tuple[str, ...] = (
    "Energy|Demand|Industry",
    "Energy|Demand|Other Sector",
    "Industrial Processes",
    "Other",
)
"""
Sectors (in the input naming convention) that make up the gridding industrial sector
"""

REQUIRED_GRIDDING_SECTORS_MODEL_REGION_INPUT: tuple[str, ...] = (
    "Energy|Supply",
    *INDUSTRIAL_SECTOR_GRIDDING_COMPONENTS_INPUT,
    "Energy|Demand|Residential and Commercial and AFOFI",
    "Product Use",
    # Technically, domestic aviation could be reported just
    # at the world level and it would be fine.
    # In practice, no-one does that and the logic is much simpler
    # if we assume it has to be reported regionally
    # (because then domestic aviation and transport are on the same regional 'grid')
    # so do that for now.
    DOMESTIC_AVIATION_SECTOR_INPUT,
    TRANSPORTATION_SECTOR_INPUT,
    "Waste",
    # Compulsory agriculture parts
    "AFOLU|Agriculture",
    "AFOLU|Agricultural Waste Burning",
    "AFOLU|Land|Fires|Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning",
)
"""
Sectors (input naming) required in the input at the model region level for gridding
"""

REQUIRED_GRIDDING_SECTORS_MODEL_REGION_INPUT_CO2_EXCEPTIONS: tuple[str, ...] = (
    "AFOLU|Land|Fires|Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning",
    "AFOLU|Land|Fires|Peat Burning",
)
"""
Sectors (input naming) at the model region level that aren't required for CO2

These are burning sectors because burning should come from the model internally
to avoid double counting.
In general, these exceptions are super messy
but we think this is the best available interpretation of everything.
"""

REQUIRED_MODEL_REGION_VARIABLES_INPUT: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_INPUT
    for sector in REQUIRED_GRIDDING_SECTORS_MODEL_REGION_INPUT
    if not (
        species == "CO2"
        and sector in REQUIRED_GRIDDING_SECTORS_MODEL_REGION_INPUT_CO2_EXCEPTIONS
    )
)

OPTIONAL_GRIDDING_SECTORS_MODEL_REGION_INPUT: tuple[str, ...] = (
    "Other Capture and Removal",
    "AFOLU|Land|Land Use and Land-Use Change",
    "AFOLU|Land|Harvested Wood Products",
    "AFOLU|Land|Other",
    "AFOLU|Land|Wetlands",
    "AFOLU|Land|Fires|Peat Burning",
)
"""
Sectors (input naming) that are optional at the model region level for gridding
"""

OPTIONAL_MODEL_REGION_VARIABLES_INPUT: tuple[str, ...] = tuple(
    f"Emissions|{species}|{sector}"
    for species in REQUIRED_GRIDDING_SPECIES_INPUT
    for sector in OPTIONAL_GRIDDING_SECTORS_MODEL_REGION_INPUT
)
"""
Optional variables at the model region level (input naming convention)
"""

ALL_MODEL_REGION_VARIABLES_INPUT: tuple[str, ...] = tuple(
    (*REQUIRED_MODEL_REGION_VARIABLES_INPUT, *OPTIONAL_MODEL_REGION_VARIABLES_INPUT)
)
"""
All variables considered at the model region level (input naming convention)
"""

INDEPENDENT_MODEL_REGION_VARIABLES_INPUT: tuple[str, ...] = tuple(
    v
    for v in ALL_MODEL_REGION_VARIABLES_INPUT
    # Avoid double counting
    if DOMESTIC_AVIATION_SECTOR_INPUT not in v
)
"""
Variables that are independent at the model region level (input naming convention)
"""
