#!/usr/bin/env bash

GIT_TAG="${GITHUB_REF##*/}"
if [[ $GIT_TAG =~ [0-9]+.[0-9]+.[0-9]+ ]]; then
    # valid tag
    odir=""
else
    # master branch
    odir="experimental/"
fi
# Upload the new bundle
curl -k --ftp-create-dirs -T SIFT_*.*.*_*.* --key ~/.ssh/id_rsa_sftp sftp://sift@ftp.ssec.wisc.edu/${odir}
# Delete any old
if [[ $GIT_TAG =~ [0-9]+.[0-9]+.[0-9]+ ]]; then
    curl -k -l --key ~/.ssh/id_rsa_sftp sftp://sift@ftp.ssec.wisc.edu/experimental/ | grep SIFT_*.*.*_*.* | xargs -I{} -- curl -k -v --key /tmp/sftp_rsa sftp://sift@ftp.ssec.wisc.edu/experimental/ -Q "RM experimental/{}"
fi
