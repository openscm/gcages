# Changelog

Versions follow [Semantic Versioning](https://semver.org/) (`<major>.<minor>.<patch>`).

Backward incompatible (breaking) changes will only be introduced in major versions
with advance notice in the **Deprecations** section of releases.

<!--
You should *NOT* be adding new changelog entries to this file,
this file is managed by towncrier.
See `changelog/README.md`.

You *may* edit previous changelogs to fix problems like typo corrections or such.
To add a new changelog entry, please see
`changelog/README.md`
and https://pip.pypa.io/en/latest/development/contributing/#news-entries,
noting that we use the `changelog` directory instead of news,
markdown instead of restructured text and use slightly different categories
from the examples given in that link.
-->

<!-- towncrier release notes start -->

## gcages v0.8.0 (2025-07-12)

### ⚠️ Breaking Changes

- Reverted the addition of a CO2 AFOLU gridding sector in CMIP7 ScenarioMIP pre-processing. ([#29](https://github.com/openscm/gcages/pull/29))


## gcages v0.7.0 (2025-07-11)

### ⚠️ Breaking Changes

- Managed the removal of subsectors for `Emissions|CO2|AFOLU` tree ([#23](https://github.com/openscm/gcages/pull/23))
- Introduced the handling of the `Carbon Removal` tree ([#25](https://github.com/openscm/gcages/pull/25))

### 🐛 Bug Fixes

- Moved in `Agriculture` input sectors previously accounted in `Agriculture Waste Burning` ([#18](https://github.com/openscm/gcages/pull/18))


## gcages v0.6.1 (2025-07-10)

This release included the changes for the v0.7.0 release,
but was released with the wrong version number.


## gcages v0.6.0 (2025-05-04)

### ⚠️ Breaking Changes

- Clarified that aneris overrides should be supplied as a [pd.Series][pandas.Series], not [pd.DataFrame][pandas.DataFrame] ([#16](https://github.com/openscm/gcages/pull/16))

### 🐛 Bug Fixes

- Fixed processing of aneris-based harmonisation overrides in the case where they are specified for some but not all scenarios (previously, the overrides would be filtered to an empty [pd.Series][pandas.Series] and then an error would occur later in the workflow) ([#16](https://github.com/openscm/gcages/pull/16))


## gcages v0.5.0 (2025-05-03)

### ⚠️ Breaking Changes

- Moved [align_history_to_data_at_time][gcages.harmonisation.common.align_history_to_data_at_time] from `gcages.harmonisation` to `gcages.harmonisation.common` ([#15](https://github.com/openscm/gcages/pull/15))

### 🆕 Features

- Added a generic harmoniser using [aneris](https://aneris.readthedocs.io/), [AnerisHarmoniser][gcages.harmonisation.AnerisHarmoniser] ([#15](https://github.com/openscm/gcages/pull/15))


## gcages v0.4.0 (2025-04-29)

### 🆕 Features

- Added pre-processing to support CMIP7's ScenarioMIP, see [How to run the CMIP7 ScenarioMIP workflow][how-to-run-the-cmip7-scenariomip-workflow] and [gcages.cmip7_scenariomip][]. ([#14](https://github.com/openscm/gcages/pull/14))


## gcages v0.3.0 (2025-04-15)

### 🆕 Features

- Added simple climate model running and post-processing as it was done in AR6, see [gcages.ar6.AR6SCMRunner][] and [gcages.ar6.AR6PostProcessor][] ([#13](https://github.com/openscm/gcages/pull/13))

### 📚 Improved Documentation

- Added a demonstration of simple climate model running and post-processing to the AR6 workflow docs ([#13](https://github.com/openscm/gcages/pull/13))


## gcages v0.2.1 (2025-04-14)

### 📚 Improved Documentation

- Added missing tutorial to the navigation bar in the docs ([#12](https://github.com/openscm/gcages/pull/12))


## gcages v0.2.0 (2025-04-14)

### ⚠️ Breaking Changes

- Updated to use `gcages` naming conventions throughout ([#10](https://github.com/openscm/gcages/pull/10))
- Simplified the renaming API, now all that is needed (and available) is [gcages.ar6.renaming.convert_variable_name] and [gcages.ar6.renaming.SupportedNamingConventions].
  Replace calls to old functions like `convert_gcages_variable_to_iamc(variable)` with `convert_variable_name(variable, from_convention=SupportedNamingConventions.GCAGES, to_convention=SupportedNamingConventions.IAMC)`. ([#11](https://github.com/openscm/gcages/pull/11))

### 🆕 Features

- Added the AR6 harmonisation module [gcages.ar6.harmonisation][gcagesar6harmonisation] and associated supporting code ([#5](https://github.com/openscm/gcages/pull/5))
- Added the [gcages.assertions][] module and [gcages.harmonisation.assert_harmonised][]. These additions provide a number of useful assertions for run-time data checks. ([#6](https://github.com/openscm/gcages/pull/6))
- Added [gcages.databases][] and conversions to OpenSCM-Runner naming conventions in [gcages.renaming][] ([#10](https://github.com/openscm/gcages/pull/10))
- - Added infilling as it was done in AR6, see [gcages.ar6.AR6Infiller][]
  - Added [gcages.completeness][] for checking that data is complete (see the module for the exact definition of "complete")
  - Added support for renaming to RCMIP and the AR6 CFC infilling database naming conventions

  ([#11](https://github.com/openscm/gcages/pull/11))

### 🎉 Improvements

- Added [gcages.typing][] to help clarify the kind of data we expect throughout. ([#6](https://github.com/openscm/gcages/pull/6))

### 📚 Improved Documentation

- Added docs on renaming and added a section on naming to the docs on how to run the AR6 workflow ([#10](https://github.com/openscm/gcages/pull/10))
- Added a demonstration of infilling to the AR6 workflow docs ([#11](https://github.com/openscm/gcages/pull/11))

### 🔧 Trivial/Internal Changes

- [#3](https://github.com/openscm/gcages/pull/3), [#4](https://github.com/openscm/gcages/pull/4), [#8](https://github.com/openscm/gcages/pull/8)


## gcages v0.1.0 (2025-02-07)

No significant changes.
