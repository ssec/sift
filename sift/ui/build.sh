#!/bin/bash
set -ex
for fn in *.ui; do 
  pyuic4 ${fn} >${fn//.ui}_ui.py
done
