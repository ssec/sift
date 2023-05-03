#!/bin/bash
set -Eeuo pipefail

notify () {
  FAILED_COMMAND="$(caller): ${BASH_COMMAND}" \
    # perform notification here
}

trap notify ERR

# force a friendly umask
umask 002

#set font config path
export FONTCONFIG_PATH=/etc/fonts

#get script path and then one directory up
SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
SIFT_HOME="$(dirname "$SCRIPT_PATH")"

echo "SIFT_HOME directory is $SIFT_HOME."

#set XRIT_DECOMPRESS_PATH
export XRIT_DECOMPRESS_PATH="$SIFT_HOME/etc/xRITDecompress/xRITDecompress"

#use perl by default. if perl is not there, default to readlink
PERL_EXISTS=`which perl`
RET_CODE=$?
if [ $RET_CODE -eq 0 ]; then
   ABSPATH="$(perl -e "use Cwd 'abs_path'; print abs_path('$0')")"
   CDIR=`dirname "$ABSPATH"`
else
   # perl doesn't exist so try to use readlink with -f option
   res_readlink="$(readlink -f $0 2>&1)"
   RET_CODE=$?
   if [ $RET_CODE -eq 0 ]; then
      CDIR=`dirname "$res_readlink"`
   else
      #try readlink without -f
      res_readlink="$(readlink $0 2>&1)"
      RET_CODE=$?
      if [ $RET_CODE -eq 0 ]; then
         CDIR=`dirname "$res_readlink"`
      else
         #do not use readlink default to a version that doesn't support symbolic link
         CDIR=`dirname "$0"`
      fi
   fi
fi


#to re-create the absolute path
HERE=$(unset CDPATH; cd "$CDIR"; pwd)
export SIFT_HOME=$(unset CDPATH; cd "$HERE/.."; pwd)

#set SIFT VARs
SIFT_LOGS="$SIFT_HOME/logs"

#run update settings file
echo "Create default settings if necessary:"
echo "--------------------------------------------"
"$SIFT_HOME"/etc/update_setting.py
echo "--------------------------------------------"
echo ""

echo "Check if there is a backup available of auto_update.yaml and catalogue.yaml to restore:"
echo "--------------------------------------------"
"$SIFT_HOME"/etc/copy_autoupdate_settings.py restore_interactive
echo "--------------------------------------------"
echo ""

cd "$SIFT_HOME"/lib
echo "Launching SIFT"

#./sift >"$SIFT_LOGS/sift.log" 2>&1
./sift
res="$?"

exit $res
