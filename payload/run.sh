#!/bin/bash
# Script assumes python venv is already properly configured
# required env vars:
# $BOT_PAT
# $WHEEL_RELEASE
# $WHEEL_BASEURL
set -ex

cat <<EOF > ${HOME}/.netrc
machine gitlab.com
login rhel-ai-wheels-prefetch-token-rhoai 
password $BOT_PAT
EOF

trap "rm -rf ${HOME}/.netrc release release.tar.gz" EXIT

# WHEEL_RELEASE="2.20.55+vllm-cuda-ubi9-x86_64"

# Gitlab project ID, etc should be static 
WHEEL_RELEASE_ARTIFACTS="https://gitlab.com/api/v4/projects/68045055/packages/generic/vllm-wheels/${WHEEL_RELEASE}/wheels-${WHEEL_RELEASE}.tar.gz"


# NOTE - ensure that flashinfer is included in wheel thing

curl --netrc -o release.tar.gz ${WHEEL_RELEASE_ARTIFACTS} 
tar zxvf release.tar.gz 
./release/install_wheels.sh

