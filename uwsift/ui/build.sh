#!/bin/bash
set -ex
for fn in *.ui; do 
  pyuic5 ${fn} >${fn//.ui}_ui.py
done
