#
# Build System to create Conda and PyInstaller Packages for SIFT
#
#
# Author: Alexander Rettig
cmake_minimum_required( VERSION 3.10 ) # May be relaxed to allow older CMake versions

# 'LANGUAGES' might be changed to Python when a Python toolchain becomes available
project( mtgsift LANGUAGES NONE )

#-------------------------------------------------------------------------------------------------
# Global Settings
#-------------------------------------------------------------------------------------------------

list( APPEND CMAKE_MODULE_PATH "cmake/" )

set( PYTHON_VERSION 3.7 )

#-------------------------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------------------------

# Building packages is only supported from within a conda environment (for now)
find_package( Conda REQUIRED )

#-------------------------------------------------------------------------------------------------
# Common Packaging Settings
#-------------------------------------------------------------------------------------------------

if( NOT ${PROJECT_NAME}_CONDA_ENVIRONMENT )
  set( ${PROJECT_NAME}_CONDA_ENVIRONMENT "$ENV{CONDA_DEFAULT_ENV}" CACHE STRING "Name (or absolute path) of conda environment in which ${PROJECT_NAME} is build/packaged" )
endif()
if( NOT "${Conda_ENVIRONMENT}" STREQUAL "${${PROJECT_NAME}_CONDA_ENVIRONMENT}" )
  message( FATAL_ERROR
    "Current Conda environment '${Conda_ENVIRONMENT}' doesn't match the configured build environment '${${PROJECT_NAME}_CONDA_ENVIRONMENT}'!\n"
    "Please run\n"
    "  conda activate ${${PROJECT_NAME}_CONDA_ENVIRONMENT}\n"
    "in your working shell or synchronize the value of ${PROJECT_NAME}_CONDA_ENVIRONMENT in the CMake configuration." )
endif()

set( ${PROJECT_NAME}_DEPENDENCIES_CHANNEL "conda-forge" CACHE STRING "Conda channel which provides dependencies of ${PROJECT_NAME}." )

#-------------------------------------------------------------------------------------------------
# Conda Packaging
#-------------------------------------------------------------------------------------------------

option( DIST_Conda "If set, a target for building installers with Conda is created" ON )
if( DIST_Conda )

set( ${PROJECT_NAME}_CONDA_RECIPE "${PROJECT_NAME}" ) # Name of the Conda recipe, i.e. the directory name to look for 'meta.yaml' in
set( ${PROJECT_NAME}_CONDA_DIST_DIR "$ENV{HOME}/conda-channels/mtgvis" CACHE STRING "Path to local conda packages directory." )

add_custom_target( conda-packages
  COMMAND ${Conda_EXECUTABLE} build -c ${${PROJECT_NAME}_DEPENDENCIES_CHANNEL} --python ${PYTHON_VERSION} --prefix-length 30 --output-folder "${${PROJECT_NAME}_CONDA_DIST_DIR}" "conda-recipe/${${PROJECT_NAME}_CONDA_RECIPE}"
  VERBATIM
  COMMENT "Building conda package for distribution to end users ..."
  )

endif( DIST_Conda )
