#
# Build System to create Conda and PyInstaller Packages for SIFT
#
#
# Author: Alexander Rettig
cmake_minimum_required( VERSION 3.10 ) # May be relaxed to allow older CMake versions but not below 3.8 because we use COMMAND_EXPAND_LISTS

# 'LANGUAGES' might be changed to Python when a Python toolchain becomes available
project( sift LANGUAGES NONE )

#-------------------------------------------------------------------------------------------------
# Global Settings
#-------------------------------------------------------------------------------------------------

list( APPEND CMAKE_MODULE_PATH "cmake/" )

set( PYTHON_VERSION 3.10 CACHE STRING "Version of Python to setup/build SIFT for." )

#-------------------------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------------------------

# Building packages is only supported from within a conda environment (for now)
find_package( Conda REQUIRED )

#-------------------------------------------------------------------------------------------------
# Common Packaging Settings
#-------------------------------------------------------------------------------------------------

if( NOT ${PROJECT_NAME}_CONDA_ENVIRONMENT )
  set( ${PROJECT_NAME}_CONDA_ENVIRONMENT "devel-${PYTHON_VERSION}" CACHE STRING "Name (or absolute path) of conda environment in which ${PROJECT_NAME} is build/packaged" )
endif()

if( "${${PROJECT_NAME}_CONDA_ENVIRONMENT}" STREQUAL "base" )
  set( ${PROJECT_NAME}_CONDA_ENVIRONMENT "devel-${PYTHON_VERSION}" CACHE STRING "Name (or absolute path) of conda environment in which ${PROJECT_NAME} is build/packaged" FORCE )
  message( FATAL_ERROR "${PROJECT_NAME} project must not be set up in a Conda 'base' environment. Resetting to sensible default." )
endif()

set( ${PROJECT_NAME}_DEPENDENCIES_CHANNEL "conda-forge" CACHE STRING "Conda channel which provides dependencies of ${PROJECT_NAME}." )

#-------------------------------------------------------------------------------------------------
# Convenience Target 'devel-bootstrap'
#-------------------------------------------------------------------------------------------------

# Support bootstrap a development environment
# Prerequisite: Anaconda3 is installed, environment (base) activated

if( ${Conda_EXECUTABLE} MATCHES .*/?mamba )
  set( XRIT_DEFAULT ON )
else()
  set( XRIT_DEFAULT OFF )
endif()
option( WITH_xRITDecompress "If set, integrate xRITDecompress in the development environment" ${XRIT_DEFAULT} )

add_custom_target( devel-bootstrap
  COMMAND ${Conda_EXECUTABLE} create  --name ${${PROJECT_NAME}_CONDA_ENVIRONMENT} --channel conda-forge --strict-channel-priority python=${PYTHON_VERSION}
  COMMAND ${Conda_EXECUTABLE} install --name ${${PROJECT_NAME}_CONDA_ENVIRONMENT} --channel conda-forge --strict-channel-priority --file conda-envs/sift-devel-minimal.pkgs

  COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:${WITH_xRITDecompress}>,env,true> ${Conda_EXECUTABLE} install --name ${${PROJECT_NAME}_CONDA_ENVIRONMENT} --channel eumetsat --strict-channel-priority public-decomp-wt
  COMMAND echo "You may want to run the following commands manually:"
  COMMAND echo "  conda activate ${${PROJECT_NAME}_CONDA_ENVIRONMENT}"
  COMMAND echo "  conda config --env --add channels conda-forge"
  COMMAND echo "  conda config --env --set channel_priority strict"
  COMMAND echo "  ${CMAKE_COMMAND} ."
  COMMAND echo "To make sure that GRIB and BUFR readers work correctly, run the following commands manually:"
  COMMAND echo "  pip install ecmwflibs"
  COMMAND echo "  pip install eccodes"
  COMMAND echo "  pip install cfgrib"
  VERBATIM
  COMMAND_EXPAND_LISTS
  COMMENT "Setting up conda environment for development ..."
  )

#-------------------------------------------------------------------------------------------------
# Check: make sure the current environment is the one the build system was set up in.
#-------------------------------------------------------------------------------------------------

if( NOT "${Conda_ENVIRONMENT}" STREQUAL "${${PROJECT_NAME}_CONDA_ENVIRONMENT}" )
  message( WARNING
    "Current Conda environment '${Conda_ENVIRONMENT}' doesn't match the configured build environment '${${PROJECT_NAME}_CONDA_ENVIRONMENT}'!\n"
    "Please run\n"
    "  ${CMAKE_MAKE_PROGRAM} devel-bootstrap\n"
    "to set up an initial development environment '${${PROJECT_NAME}_CONDA_ENVIRONMENT}'\n"
    "or if it already exists:\n"
    "  conda activate ${${PROJECT_NAME}_CONDA_ENVIRONMENT}\n"
    "in your working shell\n"
    "or synchronize the value of ${PROJECT_NAME}_CONDA_ENVIRONMENT in the CMake configuration.\n"
    )
  return()
endif()

#-------------------------------------------------------------------------------------------------
# Conda Packaging
#-------------------------------------------------------------------------------------------------

option( DIST_Conda "If set, a target for building installers with Conda is created" ON )
if( DIST_Conda )

set( ${PROJECT_NAME}_CONDA_RECIPE "${PROJECT_NAME}" ) # Name of the Conda recipe, i.e. the directory name to look for 'meta.yaml' in
set( ${PROJECT_NAME}_CONDA_DIST_DIR "$ENV{HOME}/conda-channels/uwsift" CACHE STRING "Path to local conda packages directory." )

add_custom_target( conda-packages
  COMMAND ${Conda_EXECUTABLE} build -c ${${PROJECT_NAME}_DEPENDENCIES_CHANNEL} --python ${PYTHON_VERSION} --prefix-length 30 --output-folder "${${PROJECT_NAME}_CONDA_DIST_DIR}" "conda-recipe/${${PROJECT_NAME}_CONDA_RECIPE}"
  VERBATIM
  COMMENT "Building conda package for distribution to end users ..."
  )

endif( DIST_Conda )

#-------------------------------------------------------------------------------------------------
# PyInstaller Packaging
#-------------------------------------------------------------------------------------------------

option( DIST_PyInstaller "If set, a target for building installers with PyInstaller is created" ON )
if( DIST_PyInstaller )

  find_package( PyInstaller REQUIRED )
  find_package( SelfPackers )
  if( NOT SELF_PACKER_FOR_EXECUTABLE )
    message( "PyInstaller packaging: Consider to install UPX (https://upx.github.io/) to get a smaller package." )
  endif()

#-------------------------------------------------------------------------

set( ${PROJECT_NAME}_PYINSTALLER_BASE_DIR "${CMAKE_BINARY_DIR}/../pyinstaller" CACHE STRING
  "Path to where PyInstaller shall put its artefacts.\n
   CAUTION: The directory must be outside the source tree to avoid that PyInstaller artefacts interfere with Conda packaging.\n
   WARNING: The directory is *owned* by the build system, it will be removed and recreated by the target 'pyinstaller-package'!" )
  message( WARNING "Please consider carefully: "
    " The directory configured as ${PROJECT_NAME}_PYINSTALLER_BASE_DIR='${${PROJECT_NAME}_PYINSTALLER_BASE_DIR}'"
    " is *owned* by the build system, it will be removed and recreated by the target 'pyinstaller-package'!" )

message( "Creating target 'pyinstaller-package'" )
set( _pyi_dir_ ${${PROJECT_NAME}_PYINSTALLER_BASE_DIR} )
add_custom_command(OUTPUT pyinstaller-build
  COMMAND ${CMAKE_COMMAND} -E remove_directory "${_pyi_dir_}"
  COMMAND pip install -e .
  COMMAND ${PyInstaller_EXECUTABLE} --distpath "${_pyi_dir_}/package" --workpath "${_pyi_dir_}/work" --clean -y ${PROJECT_NAME}-pyinstaller-package.spec
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_pyi_dir_}/dist/${PROJECT_NAME}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_pyi_dir_}/dist/${PROJECT_NAME}/logs"
  COMMAND ${CMAKE_COMMAND} -E rename "${_pyi_dir_}/package/${PROJECT_NAME}/resources/bin" "${_pyi_dir_}/dist/${PROJECT_NAME}/bin"
  COMMAND ${CMAKE_COMMAND} -E rename "${_pyi_dir_}/package/${PROJECT_NAME}/etc"           "${_pyi_dir_}/dist/${PROJECT_NAME}/etc"
  COMMAND ${CMAKE_COMMAND} -E rename "${_pyi_dir_}/package/${PROJECT_NAME}"               "${_pyi_dir_}/dist/${PROJECT_NAME}/lib"
  BYPRODUCTS "${_pyi_dir_}/dist/${PROJECT_NAME}/lib/${PROJECT_NAME}"
  VERBATIM
  COMMENT "Building PyInstaller package in '${_pyi_dir_}/dist/${PROJECT_NAME}/lib'"
  )
add_custom_target( pyinstaller-package
  COMMAND ${CMAKE_COMMAND} -E tar "cf" "${PROJECT_NAME}.zip" --format=zip "${PROJECT_NAME}"
  DEPENDS pyinstaller-build
  BYPRODUCTS "${_pyi_dir_}/dist/${PROJECT_NAME}.zip"
  WORKING_DIRECTORY "${_pyi_dir_}/dist"
  VERBATIM
  COMMENT "Package PyInstaller output for SIFT"
)

endif( DIST_PyInstaller )
