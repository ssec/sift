#!/usr/bin/env sh

set -e
set -u

PROJ_NAME='sift'

BASE_DIR="$(dirname -- "$0")/.."
cd "$BASE_DIR" || exit
BASE_DIR="$(pwd -P)"

if [ -d "$BASE_DIR/lib/$PROJ_NAME" ] || [ ! -x "$BASE_DIR/lib/$PROJ_NAME" ]; then
  echo "Missing binary @ '$BASE_DIR/lib/$PROJ_NAME' to start $PROJ_NAME."
  exit 1
fi

PATH="$BASE_DIR/bin:$PATH"
export PATH

cd "$BASE_DIR/lib" || exit

"./$PROJ_NAME" "$@"
