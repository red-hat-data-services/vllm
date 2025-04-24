variable "REPOSITORY" {
  default = "quay.io/vllm/vllm"
}

# GITHUB_* variables are only available in github actions
variable "GITHUB_SHA" {}
variable "GITHUB_REPOSITORY" {}
variable "GITHUB_RUN_ID" {}

variable "VLLM_VERSION" {} # set by github actions or manually?

target "docker-metadata-action" {} // populated by gha docker/metadata-action

target "_common" {
  context = "."

  args = {
    BASE_UBI_IMAGE_TAG = "9.5-1736404155"
    PYTHON_VERSION = "3.12"
  }

  inherits = ["docker-metadata-action"]

  platforms = [
    "linux/amd64",
  ]
  labels = {
    "org.opencontainers.image.source" = "https://github.com/${GITHUB_REPOSITORY}"
    "vcs-ref" = "${GITHUB_SHA}"
    "vcs-type" = "git"
  }
}

group "default" {
  targets = [
    "cuda",
  ]
}

target "cuda" {
  inherits = ["_common"]
  dockerfile = "Dockerfile.ubi"

  args = {
    BASE_UBI_IMAGE_TAG = "9.5-1739420147"
    PYTHON_VERSION = "3.12"
    # CUDA_VERSION = "12.4" # TODO: the dockerfile cannot consume the cuda version
    LIBSODIUM_VERSION = "1.0.20"
    VLLM_TGIS_ADAPTER_VERSION = "0.7.0"

    FLASHINFER_VERSION = "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.1.post2/flashinfer_python-0.2.1.post2+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
  }

  tags = [
    "${REPOSITORY}:${replace(VLLM_VERSION, "+", "_")}", # vllm_version might contain local version specifiers (+) which are not valid tags
    "${REPOSITORY}:${GITHUB_SHA}",
    "${REPOSITORY}:${formatdate("YYYY-MM-DD-hh-mm", timestamp())}"
  ]
}
