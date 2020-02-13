# Find PyInstaller
# Will find the path to Makespec.py and Build.py

# python Makespec.py [opts] <scriptname> [<scriptname> ...]
# python Build.py specfile
find_program( PyInstaller_EXECUTABLE
  NAMES pyinstaller
  DOC "Path to the pyinstaller executable"
  )

# $ python Makespec.py hello.py
# -> wrote /home/mmalaterre/Projects/pyinstaller/hello/hello.spec
set( PyInstaller_MAKESPEC
  ${PyInstaller_PATH}/Makespec.py
  )

set( PyInstaller_BUILD_SPEC
  ${PyInstaller_PATH}/Build.py
  )

# Look for Python:
#find_package( PythonLibs REQUIRED )
