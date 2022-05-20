#!/bin/bash
set -x

HOME_PYINSTALLER_BUILD="../../pyinstaller/onedir/"
DIST_OUTPUT="../../distribution-build"

SCR_TEMPLATE="./mtgsift-dist-template"

rm -Rf $DIST_OUTPUT/
mkdir -p $DIST_OUTPUT

#copy the dist template
cp -R "$SCR_TEMPLATE" "$DIST_OUTPUT/mtgsift-dist"

#copy the built pyinstaller version as lib in the dist
cp -R "$HOME_PYINSTALLER_BUILD/dist/mtgsift" "$DIST_OUTPUT/mtgsift-dist/lib"

#move out the libraries creating some dependencies conflicts (keep them for the moment)
mv "$DIST_OUTPUT/mtgsift-dist/lib/libudev.so.1" "$DIST_OUTPUT/mtgsift-dist/etc/extra-libs"
mv "$DIST_OUTPUT/mtgsift-dist/lib/libxcb-shm.so.0" "$DIST_OUTPUT/mtgsift-dist/etc/extra-libs"

#Adapt Qt5 if necessary
PYQT5LIB="$DIST_OUTPUT/mtgsift-dist/lib/PyQt5/Qt"
#copy icu file
cp -R $PYQT5LIB/resources/icudtl.dat $PYQT5LIB/libexec

cd $PYQT5LIB/libexec
ln -fs ../resources/qtwebengine_resources_100p.pak qtwebengine_resources_100p.pak
ln -fs ../resources/qtwebengine_resources_200p.pak qtwebengine_resources_200p.pak
ln -fs ../resources/qtwebengine_resources.pak qtwebengine_resources.pak
ln -fs ../translations/qtwebengine_locales qtwebengine_locales
