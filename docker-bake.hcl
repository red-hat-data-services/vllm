variable "REPOSITORY" {
  default = "quay.io/vllm/vllm"
}

# GITHUB_* variables are set as env vars in github actions
variable "GITHUB_SHA" {}
variable "GITHUB_REPOSITORY" {}
variable "GITHUB_RUN_ID" {}

variable "VLLM_VERSION" {}

variable "PYTHON_VERSION" {
  default = "3.12"
}

variable "ROCM_VERSION" {
  default = "6.3.4"
}

variable "VLLM_TGIS_ADAPTER_VERSION" {
  default = "0.7.0"
}


target "docker-metadata-action" {} // populated by gha docker/metadata-action

target "_common" {
  context = "."

  args = {
    BASE_UBI_IMAGE_TAG = "9.5-1742914212"
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
    "rocm",
  ]
}

target "cuda" {
  inherits = ["_common"]
  dockerfile = "Dockerfile.ubi"

  args = {
    PYTHON_VERSION = "${PYTHON_VERSION}"
    # CUDA_VERSION = "12.4" # TODO: the dockerfile cannot consume the cuda version
    LIBSODIUM_VERSION = "1.0.20"
    VLLM_TGIS_ADAPTER_VERSION = "${VLLM_TGIS_ADAPTER_VERSION}"

    FLASHINFER_VERSION = "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.1.post2/flashinfer_python-0.2.1.post2+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
  }

  tags = [
    "${REPOSITORY}:${replace(VLLM_VERSION, "+", "_")}", # vllm_version might contain local version specifiers (+) which are not valid tags
    "${REPOSITORY}:${GITHUB_SHA}",
    "${REPOSITORY}:${formatdate("YYYY-MM-DD-hh-mm", timestamp())}"
  ]
}

target "rocm" {
  inherits = ["_common"]
  dockerfile = "Dockerfile.rocm.ubi"

  args = {
    PYTHON_VERSION = "${PYTHON_VERSION}"
    ROCM_VERSION = "${ROCM_VERSION}"
    LIBSODIUM_VERSION = "1.0.20"
    VLLM_TGIS_ADAPTER_VERSION = "${VLLM_TGIS_ADAPTER_VERSION}"
  }

  tags = [
    "${REPOSITORY}:${replace(VLLM_VERSION, "+", "_")}", # vllm_version might contain local version specifiers (+) which are not valid tags
    "${REPOSITORY}:${GITHUB_SHA}",
    "${REPOSITORY}:${formatdate("YYYY-MM-DD-hh-mm", timestamp())}"
  ]
}
