"""
Integration tests of our pre-processing for CMIP7 ScenarioMIP
"""

# Tests to write:
#
# - checking reporting and internal consitency before renaming and shuffling
#   - just do this with nomenclature
#   - wrap the error message as best you can
# - shuffling transport info around
#   - make sure that the transport is changed as expected
#     - Add domestic and international aviation together to create a new variable
#     - Remove domestic aviation from the transportation variable
#       and give this result a new name too
#   - nothing else should change
# - renaming things to how it needs to be for harmonisation
#   - CEDS namings (which we want in the end) are:
#   -  "Energy Sector",
#   -  "International Shipping",
#   -  "Residential Commercial Other"
#   -  "Solvents Production and Application"
#   -  "Agriculture"
#   -  "Agricultural Waste Burning",
#   -  "Forest Burning",
#   -  "Grassland Burning",
#   -  "Peat Burning",
# - some regression test of processing something (probably a salted REMIND output)
#
# Underlying logic:
# - we're doing region-sector harmonisation
# - hence we need regions and sectors lined up very specifically with CEDS
# - hence this kind of pre-processing
#   (if you want different pre-processing, use something more like AR6)
