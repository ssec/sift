find_program( Conda_EXECUTABLE
  NAMES mamba conda
  PATHS ${CONDA_PREFIX}/bin ${CONDA_PREFIX}/Scripts ENV PATH
  )

if( NOT DEFINED ${Conda_EXECUTABLE} AND DEFINED ENV{CONDA_EXE} )
  set( Conda_EXECUTABLE $ENV{CONDA_EXE} CACHE PATH "Path to an executable" )
endif()

if( DEFINED ENV{CONDA_PREFIX} )
  set( CONDA_PREFIX $ENV{CONDA_PREFIX} )
else()
  # Assuming the active conda environment is on PATH, this finds the path of bin/ in the environment
  if( Conda_EXECUTABLE )
    execute_process( COMMAND ${Conda_EXECUTABLE} info --root
      OUTPUT_VARIABLE CONDA_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    execute_process(COMMAND ${Conda_EXECUTABLE} info --envs )
  else()
    message( "No Conda installation found in PATH!\nPATH=$ENV{PATH}\n" )
  endif()
endif()

set( Conda_ENVIRONMENT_DIR ${CONDA_PREFIX} )
set( Conda_ENVIRONMENT     $ENV{CONDA_DEFAULT_ENV} )

#-------------------------------------------------------------------------------
# Determine Conda repository platform name
#
# Asking conda for that information is tedious, let's calculate ourselves

string( TOLOWER ${CMAKE_SYSTEM_NAME} Conda_SYSTEM_NAME )
if( ${CMAKE_SYSTEM_PROCESSOR} MATCHES .*64.* )
  set( Conda_SYSTEM_PROCESSOR 64 )
else()
  set( Conda_SYSTEM_PROCESSOR 32 )
endif()

set( Conda_PLATFORM ${Conda_SYSTEM_NAME}-${Conda_SYSTEM_PROCESSOR} )

#-------------------------------------------------------------------------------
# Finalize

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Conda
  REQUIRED_VARS Conda_ENVIRONMENT Conda_EXECUTABLE Conda_PLATFORM
  )
