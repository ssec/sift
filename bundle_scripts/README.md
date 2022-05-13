# Bundle Scripts

The scripts in this directory are specially constructed to run from a
conda-pack'd bundled installation of SIFT. The types of scripts included are
currently:

1. `SIFT.X` where `X` corresponds to a scripting extension specific to each
   platform. This is `.sh` for Linux (CentOS 7+), `.command` for OSX, and
   `.bat` for Windows. These scripts are placed in the root directory of
   the released bundle.

Note to reuse code as much as possible some scripts may be copied to
the appropriate name rather than existing as separate files.
