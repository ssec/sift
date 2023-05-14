#!/bin/bash
set -ex

ui_files=${@:-*.ui}

for fn in ${ui_files[@]}; do
  fn="${fn%.ui}.ui"  # enforce file extension '.ui'
  pyuic5 "${fn}" >"${fn%.ui}_ui.py"
done
