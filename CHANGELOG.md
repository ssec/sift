## Version 2.0.0b1 (2024/09/04)

### Issues Closed

* [Issue 380](https://github.com/ssec/sift/issues/380) - Allow setting of satpy config path via env variable
* [Issue 376](https://github.com/ssec/sift/issues/376) - MTG FCI FDSS 500m does not work with LEO data
* [Issue 372](https://github.com/ssec/sift/issues/372) - "Export image"-functionality does not know how to deal with .jpg/.png formats? ([PR 375](https://github.com/ssec/sift/pull/375) by [@djhoese](https://github.com/djhoese))
* [Issue 276](https://github.com/ssec/sift/issues/276) - 'del' key to remove layers in 1.1.1 crashes SIFT if no layer is selected
* [Issue 118](https://github.com/ssec/sift/issues/118) - Column sorting in open/import/cache dialog
* [Issue 14](https://github.com/ssec/sift/issues/14) - imageio now requires separate imageio-ffmpeg package
* [Issue 13](https://github.com/ssec/sift/issues/13) - Data probing not accurate
* [Issue 9](https://github.com/ssec/sift/issues/9) - Overview image of data layers are misaligned
* [Issue 7](https://github.com/ssec/sift/issues/7) - Loading / Displaying pre-defined RGB'S from Satpy

In this release 9 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 428](https://github.com/ssec/sift/pull/428) - Fix resolution formatting for numpy floats
* [PR 382](https://github.com/ssec/sift/pull/382) - Fix usage of deprecated Pillow "textsize" method
* [PR 375](https://github.com/ssec/sift/pull/375) - Fix export image and switch to imageio v3 API ([372](https://github.com/ssec/sift/issues/372))

#### Features added

* [PR 422](https://github.com/ssec/sift/pull/422) - Fix MTG-LI filepattern filter and add FCI L2 BUFR/GRIB readers to list
* [PR 381](https://github.com/ssec/sift/pull/381) - Fix setting of external Satpy component configuration

#### Documentation changes

* [PR 428](https://github.com/ssec/sift/pull/428) - Fix resolution formatting for numpy floats
* [PR 367](https://github.com/ssec/sift/pull/367) - Update readme for install and contributing docs

In this release 7 pull requests were closed.


## Version 2.0.0b0 (2023/05/25)

Much of the work for this version was done by a private contractor in a non-GitHub
git service. As such, the normal GitHub issue/PR-based changelog is not available.
Below is a summary of the changes.

- reading of data from both geostationary (GEO) as well as low-Earth-orbit (LEO)
  satellite instruments
- visualization of point data (e.g. lightning)
- support for composite (RGB) visualization
- an improved timeline manager
- integration of a statistics module
- full resampling functionalities using Pyresample
- an automatic update/monitoring mode
- partial redesign of the UI/UX
- ... many more small but useful features!

## Version 1.2.3 (2022/02/04)

### Issues Closed

* [Issue 305](https://github.com/ssec/sift/issues/305) - uwsift 1.1.3 doesn't work w/ satpy 0.23

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 331](https://github.com/ssec/sift/pull/331) - Fix Python 3.10 incompatibilities

In this release 1 pull request was closed.


## Version 1.2.2 (2021/10/29)

### Pull Requests Merged

#### Bugs fixed

* [PR 330](https://github.com/ssec/sift/pull/330) - Fix line edits changing to unwanted value

In this release 1 pull request was closed.


## Version 1.2.1 (2021/10/28)

### Pull Requests Merged

#### Bugs fixed

* [PR 329](https://github.com/ssec/sift/pull/329) - Fix compatibility with vispy 0.8.0+

In this release 1 pull request was closed.


## Version 1.2.0 (2021/09/18)

### Issues Closed

* [Issue 325](https://github.com/ssec/sift/issues/325) - GOES-18 ([PR 326](https://github.com/ssec/sift/pull/326) by [@djhoese](https://github.com/djhoese))
* [Issue 323](https://github.com/ssec/sift/issues/323) - Color limit sliders only respond to direct click events ([PR 324](https://github.com/ssec/sift/pull/324) by [@djhoese](https://github.com/djhoese))

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 326](https://github.com/ssec/sift/pull/326) - Fix GOES-18 platform not being recognized properly ([325](https://github.com/ssec/sift/issues/325))
* [PR 319](https://github.com/ssec/sift/pull/319) - Fix half pixel offset for point probing
* [PR 317](https://github.com/ssec/sift/pull/317) - Fix validity check of reprojected mesh in TiledGelocatedImageVisual
* [PR 316](https://github.com/ssec/sift/pull/316) - Improve log message when tiled image isn't displayed

#### Features added

* [PR 324](https://github.com/ssec/sift/pull/324) - Update color limit sliders to update display instantly ([323](https://github.com/ssec/sift/issues/323))
* [PR 318](https://github.com/ssec/sift/pull/318) - Remove background/overview image tile
* [PR 312](https://github.com/ssec/sift/pull/312) - Update to work with VisPy 0.7

In this release 7 pull requests were closed.


## Version 1.1.6 (2021/01/11)

### Pull Requests Merged

#### Bugs fixed

* [PR 311](https://github.com/ssec/sift/pull/311) - Retain forward slash before Libary in prefix

In this release 1 pull request was closed.


## Version 1.1.5 (2021/01/11)

### Pull Requests Merged

#### Bugs fixed

* [PR 310](https://github.com/ssec/sift/pull/310) - Replace / with \\ throughout Prefix where present in qt.conf, to fix …
* [PR 309](https://github.com/ssec/sift/pull/309) - Fix importing data with newer versions of Satpy

In this release 2 pull requests were closed.


## Version 1.1.4 (2021/01/07)

### Issues Closed

* [Issue 297](https://github.com/ssec/sift/issues/297) - SIFT 1.1.3 windows experimental builds do not start ([PR 302](https://github.com/ssec/sift/pull/302))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 307](https://github.com/ssec/sift/pull/307) - Fix issues related with newer versions of Satpy and Pyyaml
* [PR 301](https://github.com/ssec/sift/pull/301) - Add support for Satpy 0.23+ and PROJ 6.0+

In this release 2 pull requests were closed.


## Version 1.1.3 (2020/06/12)

### Pull Requests Merged

#### Bugs fixed

* [PR 296](https://github.com/ssec/sift/pull/296) - Update bundle scripts to handle installation directory being moved
* [PR 293](https://github.com/ssec/sift/pull/293) - Fix HiDPI setting being set too late
* [PR 292](https://github.com/ssec/sift/pull/292) - Fix bundle script having wrong permissions and activation path
* [PR 288](https://github.com/ssec/sift/pull/288) - Fix STANDARD_NAME not being set correctly

#### Features added

* [PR 282](https://github.com/ssec/sift/pull/282) - Add SIFT script to python package installation

In this release 5 pull requests were closed.


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
