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

#---------- Patching Shapely.geos for PyInstaller ------------------------
# This block may be removed as soon as Shapely cleanly can by PyInstaller'ed
# from within a Conda environment

find_package( Patch )

execute_process(
  COMMAND python -c "import shapely.geos; print(shapely.geos.__file__)"
  OUTPUT_VARIABLE shapely.geos.__file__
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
set( shapely.geos.patch "${CMAKE_SOURCE_DIR}/shapely.geos.patch" )

if( Patch_FOUND )
  add_custom_target( patch-shapely
    COMMAND ${Patch_EXECUTABLE} --posix --input=${shapely.geos.patch} --force --reject-file=- ${shapely.geos.__file__}
    VERBATIM
    COMMENT "Patching ${shapely.geos.__file__} for PyInstaller.
         This may fail, e.g. if the patch was already applied.
         In that case, verify '${shapely.geos.__file__}' against '${shapely.geos.patch}' please."
    )
  message( "The following file from shapely may need to be patched to work for PyInstaller:\n"
    "  '${shapely.geos.__file__}'.\n"
    "  The patch to be applied is '${shapely.geos.patch}'.\n"
    "  Please build the target 'patch-shapely' once before 'pyinstaller-package'." )
else()
  message( WARNING "The following file from shapely may need to be patched to work for PyInstaller:"
    " '${shapely.geos.__file__}'. The patch to be applied is '${shapely.geos.patch}',"
    " but there is no 'Patch' tool found/configured. You may patch the file manually." )
endif()


#-------------------------------------------------------------------------

set( ${PROJECT_NAME}_PYINSTALLER_BASE_DIR "${CMAKE_BINARY_DIR}/../pyinstaller" CACHE STRING
  "Path to where PyInstaller shall put its artefacts.\n
   CAUTION: The directory must be outside the source tree to avoid that PyInstaller artefacts interfere with Conda packaging.\n
   WARNING: The directory is *owned* by the build system, it will be removed and recreated by the target 'pyinstaller-package'!" )
  message( WARNING "Please consider carefully: "
    " The directory configured as ${PROJECT_NAME}_PYINSTALLER_BASE_DIR='${${PROJECT_NAME}_PYINSTALLER_BASE_DIR}'"
    " is *owned* by the build system, it will be removed and recreated by the target 'pyinstaller-package'!" )

foreach( PYINSTALLER_MODE onedir onefile )
  message( "Creating target 'pyinstaller-${PYINSTALLER_MODE}-package'" )
  set( _pyi_dir_ ${${PROJECT_NAME}_PYINSTALLER_BASE_DIR}/${PYINSTALLER_MODE}  )
  add_custom_target( pyinstaller-${PYINSTALLER_MODE}-package
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${${PROJECT_NAME}_PYINSTALLER_BASE_DIR}/${PYINSTALLER_MODE}
    COMMAND pip install -e .
    COMMAND ${PyInstaller_EXECUTABLE} --distpath "${_pyi_dir_}/dist" --workpath "${_pyi_dir_}/work" --clean -y ${PROJECT_NAME}-${PYINSTALLER_MODE}.spec
    BYPRODUCTS "${_pyi_dir_}/dist/${PROJECT_NAME}/${PROJECT_NAME}"
    VERBATIM
    COMMENT "Building PyInstaller package in '${_pyi_dir_}/dist/${PROJECT_NAME}/${PROJECT_NAME}'"
    )

endforeach()

endif( DIST_PyInstaller )

