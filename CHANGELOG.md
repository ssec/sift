## Version 1.1.2 (2020/01/10)

### Issues Closed

* [Issue 278](https://github.com/ssec/sift/issues/278) - Bands of different resolutions are not grouped together ([PR 279](https://github.com/ssec/sift/pull/279))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 279](https://github.com/ssec/sift/pull/279) - Fix incorrectly grouping layers from the same scene ([278](https://github.com/ssec/sift/issues/278))
* [PR 275](https://github.com/ssec/sift/pull/275) - Recognize GEO-KOMPSAT-2A satellite and LI and GLM instruments

In this release 2 pull requests were closed.


## Version 1.1.1 (2019/12/06)

### Issues Closed

* [Issue 267](https://github.com/ssec/sift/issues/267) - "Layer details"-tab shows incorrect central wavelength ([PR 268](https://github.com/ssec/sift/pull/268))
* [Issue 266](https://github.com/ssec/sift/issues/266) - Proj4 returns `inf` for antimeridian in `begin_import_products` ([PR 272](https://github.com/ssec/sift/pull/272))
* [Issue 265](https://github.com/ssec/sift/issues/265) - `_pretty_identifiers` crash if `resolution` is a float ([PR 270](https://github.com/ssec/sift/pull/270))
* [Issue 251](https://github.com/ssec/sift/issues/251) - conda installation instructions do not work

In this release 4 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 272](https://github.com/ssec/sift/pull/272) - Fix crash during importing when pyproj 2.4.2 is installed ([266](https://github.com/ssec/sift/issues/266))
* [PR 270](https://github.com/ssec/sift/pull/270) - Fix open file wizard when resolution is a floating point number ([265](https://github.com/ssec/sift/issues/265))
* [PR 269](https://github.com/ssec/sift/pull/269) - Fix grib reader not being included in available readers by default
* [PR 268](https://github.com/ssec/sift/pull/268) - Fix incorrect wavelength being taken from Satpy ([267](https://github.com/ssec/sift/issues/267))

#### Documentation changes

* [PR 273](https://github.com/ssec/sift/pull/273) - Add test_requires to setup.py

In this release 5 pull requests were closed.


## Version 1.1.0 (2019/12/04)

### Issues Closed

* [Issue 253](https://github.com/ssec/sift/issues/253) - AttributeError and core dump when removing file from selection dialogue
* [Issue 252](https://github.com/ssec/sift/issues/252) - Incorrect geolocation for ABI L1B RadC (CONUS) data ([PR 254](https://github.com/ssec/sift/pull/254))
* [Issue 243](https://github.com/ssec/sift/issues/243) - RGB "gamma" values, worked with the inverse
* [Issue 235](https://github.com/ssec/sift/issues/235) - Export image screenshots whole window ([PR 238](https://github.com/ssec/sift/pull/238))
* [Issue 234](https://github.com/ssec/sift/issues/234) - Fix numba warnings when opening SIFT ([PR 237](https://github.com/ssec/sift/pull/237))
* [Issue 233](https://github.com/ssec/sift/issues/233) - PyQt5 from conda-forge no longer builds with WebKit ([PR 236](https://github.com/ssec/sift/pull/236))
* [Issue 229](https://github.com/ssec/sift/issues/229) - Update workspace directory on Windows installations
* [Issue 227](https://github.com/ssec/sift/issues/227) - Rename package name for conda-forge and PyPI ([PR 230](https://github.com/ssec/sift/pull/230))
* [Issue 220](https://github.com/ssec/sift/issues/220) - Migrate to PyQt5 ([PR 222](https://github.com/ssec/sift/pull/222))
* [Issue 216](https://github.com/ssec/sift/issues/216) - Refactor names and modules to be more PEP8 compliant
* [Issue 210](https://github.com/ssec/sift/issues/210) - Add option for color bar on exported images ([PR 238](https://github.com/ssec/sift/pull/238))
* [Issue 3](https://github.com/ssec/sift/issues/3) - Satpy readers only version of SIFT

In this release 12 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 254](https://github.com/ssec/sift/pull/254) - Fix vertex coordinates being calculated incorrectly ([252](https://github.com/ssec/sift/issues/252))
* [PR 241](https://github.com/ssec/sift/pull/241) - Fix probing RGB layers crashes

#### Features added

* [PR 263](https://github.com/ssec/sift/pull/263) - Add caching to satpy available readers
* [PR 262](https://github.com/ssec/sift/pull/262) - Remove timeline from user interface until feature is complete
* [PR 249](https://github.com/ssec/sift/pull/249) - Add experimental global configuration object using the donfig package
* [PR 238](https://github.com/ssec/sift/pull/238) - Add colorbar option for saving images ([235](https://github.com/ssec/sift/issues/235), [210](https://github.com/ssec/sift/issues/210))
* [PR 236](https://github.com/ssec/sift/pull/236) - Use WebEngine instead of building with WebKit ([233](https://github.com/ssec/sift/issues/233))
* [PR 232](https://github.com/ssec/sift/pull/232) - Transition to Satpy for all data reading
* [PR 225](https://github.com/ssec/sift/pull/225) - Add settings for better HiDPI support
* [PR 222](https://github.com/ssec/sift/pull/222) - Migrate to PyQt5 ([220](https://github.com/ssec/sift/issues/220))

#### Documentation changes

* [PR 257](https://github.com/ssec/sift/pull/257) - Rewrite README and fix various docstring issues
* [PR 226](https://github.com/ssec/sift/pull/226) - Add AUTHORS list

#### Backwards incompatible changes

* [PR 247](https://github.com/ssec/sift/pull/247) - Remove unnecessary band metadata
* [PR 236](https://github.com/ssec/sift/pull/236) - Use WebEngine instead of building with WebKit ([233](https://github.com/ssec/sift/issues/233))
* [PR 228](https://github.com/ssec/sift/pull/228) - Move sift.control.layer_info to sift.view.layer_details
* [PR 222](https://github.com/ssec/sift/pull/222) - Migrate to PyQt5 ([220](https://github.com/ssec/sift/issues/220))
* [PR 217](https://github.com/ssec/sift/pull/217) - Refactor code to be more PEP8 compliant

In this release 17 pull requests were closed.


## Version 1.0.6 (2019/03/07)

### Issues Closed

* [Issue 218](https://github.com/ssec/sift/issues/218) - Layers not shown in RGB selection in version 1.0.5 ([PR 219](https://github.com/ssec/sift/pull/219))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 219](https://github.com/ssec/sift/pull/219) - Fix new layer families not being considered new ([218](https://github.com/ssec/sift/issues/218))
* [PR 213](https://github.com/ssec/sift/pull/213) - Fix conda recipe to work with imageio >2.5.0

#### Documentation changes

* [PR 215](https://github.com/ssec/sift/pull/215) - Rewrite initial sphinx documentation
* [PR 214](https://github.com/ssec/sift/pull/214) - Fix sphinx docs configuration to work with readthedocs

In this release 4 pull requests were closed.


## Version 1.0.5 (2019/02/27)

### Issues Closed

* [Issue 209](https://github.com/ssec/sift/issues/209) - Apply configured color enhancement to band images loaded after initial set

In this release 1 issue were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 16](https://github.com/ssec/sift/pull/16) - Fix presentation for new layers not matching similar layers

In this release 1 pull request was closed.


## Version 1.0.4 (2018/12/04)


## Version 1.0.3 (2018/11/20)


## Version 1.0.1 (2018/10/29)


## Version 1.0.0 (2018/10/25)

