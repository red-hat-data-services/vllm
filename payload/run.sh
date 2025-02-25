#!/bin/bash
# required env vars:
# $BOT_PAT
# $WHEEL_RELEASE_ARTIFACTS
# optional:
# $VLLM_TGIS_ADAPTER_VERSION
# $VLLM_WHEEL_VERSION
set -ex

cat <<EOF > ${HOME}/.netrc
machine gitlab.com
login ckodama
password $BOT_PAT
EOF

# https://docs.astral.sh/uv/configuration/indexes/#searching-across-multiple-indexes
# This will prefer to use the custom index, and fall back to pypi if needed
export UV_EXTRA_INDEX_URL=${VLLM_WHEEL_INDEX}
export UV_INDEX_STRATEGY=unsafe-first-match 

vllm="vllm[tensorizer,audio,video]"

if [[ -n "$VLLM_TGIS_ADAPTER_VERSION" ]]; then
    vllm_tgis_adapter="vllm-tgis-adapter==${VLLM_TGIS_ADAPTER_VERSION}"
fi

if [[ -n "$VLLM_WHEEL_VERSION" ]]; then
    vllm="${vllm}==${$VLLM_WHEEL_VERSION}"
fi

uv pip install $vllm $vllm_tgis_adapter

rm ${HOME}/.netrc

