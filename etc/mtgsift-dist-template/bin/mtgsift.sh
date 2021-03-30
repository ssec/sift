#!/bin/bash

# force a friendly umask
umask 002

#set font config path
export FONTCONFIG_PATH=/etc/fonts

MTGSIFT_HOME="/opt/mtg-sift/mtgsift-0.8"

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
"$MTGSIFT_HOME"/etc/update_setting.py

cd "$MTGSIFT_HOME"/lib

echo "Launching MTGSift"

#./mtgsift >"$MTGSIFT_LOGS/mtgsift.log" 2>&1
./mtgsift 
res="$?"

exit $res
