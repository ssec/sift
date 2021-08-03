#!/usr/bin/env bash

set -ex
GIT_TAG="${GITHUB_REF##*/}"

if [[ $GIT_TAG =~ [0-9]+.[0-9]+.[0-9]+ ]]; then
    # valid tag (use default script options)
    oflag=""
else
    # master branch
    version=$(python -c "from uwsift import __version__; print(__version__)")

    if [[ "${OS}" == "windows-latest" ]]; then
        ext="zip"
        platform="windows"
    else
        ext="tar.gz"
        if [[ "${OS}" == "macos-latest" ]]; then
            platform="darwin"
        else
            platform="linux"
        fi;
    fi;
    oflag="-o SIFT_${version}dev_${platform}_$(date +%Y%m%d_%H%M%S).${ext}"
fi

python build_conda_pack.py -j -1 $oflag
ls -l

set +ex
