#!/usr/bin/env bash
# Usage: SIFT.sh <command line arguments>
# Description: Initialize the SIFT installation if necessary and run SIFT
set -e

# get current base directory for this script
SOURCE="${BASH_SOURCE[0]}"
while [[ -h "$SOURCE" ]] ; do SOURCE="$(readlink "$SOURCE")"; done
BASE="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

# Remove user environment variables that may conflict with installation
unset LD_PRELOAD
unset LD_LIBRARY_PATH
unset DYLD_LIBRARY_PATH
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Activate the conda-pack'd environment
source $BASE/bin/activate

# Check if we already ran conda-unpack
install_signal="${BASE}/.installed"
if [[ "$(head -n 1 ${install_signal} 2>/dev/null)" != "${BASE}" ]]; then
    echo "Running one-time initialization of SIFT installation..."
    conda-unpack
    echo "${BASE}" > "${install_signal}"
    echo "Running SIFT..."
fi

python -m uwsift "$@"
