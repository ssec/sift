#!/bin/bash
set -Eeuo pipefail

notify () {
  FAILED_COMMAND="$(caller): ${BASH_COMMAND}" \
    # perform notification here
}

trap notify ERR

function restore_backup {
     echo ""
     echo "Restoring the default auto_update.yaml and catalogue.yaml settings:"
     echo "--------------------------------------------"
     "$MTGSIFT_HOME"/etc/copy_autoupdate_settings.py restore
     echo "--------------------------------------------"
     echo ""
}

# Call the restore_backup function
trap restore_backup EXIT

# force a friendly umask
umask 002

#set font config path
export FONTCONFIG_PATH=/etc/fonts

#get script path and then one directory up
SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MTGSIFT_HOME="$(dirname "$SCRIPT_PATH")"

echo "MTGSIFT_HOME directory is $MTGSIFT_HOME."

#set XRIT_DECOMPRESS_PATH
export XRIT_DECOMPRESS_PATH="$MTGSIFT_HOME/etc/xRITDecompress/xRITDecompress"

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
export MTGSIFT_HOME=$(unset CDPATH; cd "$HERE/.."; pwd)

#set MTGSIFT VARs
#MTGSIFT_HOME="/home/gmv/Dev/pyinstaller/distrib"
#MTGSIFT_HOME="/tcenas/home/gaubert/mtg-sift/mtgsift-distrib"
MTGSIFT_LOGS="$MTGSIFT_HOME/logs"


#run update settings file
echo "Create default settings if necessary:"
echo "--------------------------------------------"
"$MTGSIFT_HOME"/etc/update_setting.py
echo "--------------------------------------------"
echo ""

#run the back of file before setting the auto_update
echo "Backuping the default auto_update.yaml and catalogue.yaml settings:"
echo "--------------------------------------------"
"$MTGSIFT_HOME"/etc/copy_autoupdate_settings.py update
echo "--------------------------------------------"
echo ""

echo "Launching MTGSift in AUTO UPDATE MODE"

cd "$MTGSIFT_HOME"/lib
#./mtgsift >"$MTGSIFT_LOGS/mtgsift.log" 2>&1
./mtgsift
res="$?"


exit $res
