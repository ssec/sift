#!/usr/bin/env bash

set -ex

GIT_TAG="${GITHUB_REF##*/}"
if [[ $GIT_TAG =~ [0-9]+.[0-9]+.[0-9]+ ]]; then
    # valid tag
    odir=""
else
    # master branch
    odir="experimental/"
fi
# Upload the new bundle
curl -k --ftp-create-dirs -T SIFT_*.*.*_*.* --key $HOME/.ssh/id_rsa_sftp sftp://sift@ftp.ssec.wisc.edu/${odir}
set +e
# Delete any old
if [[ $GIT_TAG =~ [0-9]+.[0-9]+.[0-9]+ ]]; then
    curl -k -l --key $HOME/.ssh/id_rsa_sftp sftp://sift@ftp.ssec.wisc.edu/experimental/ | grep SIFT_*.*.*_*.* | xargs -I{} -- curl -k -v --key $HOME/.ssh/id_rsa_sftp sftp://sift@ftp.ssec.wisc.edu/experimental/ -Q "RM experimental/{}"
    if [ $? -ne 0 ]; then
        echo "Failed to delete old experimental SIFT tarballs from FTP server"
    fi
fi

set +x
