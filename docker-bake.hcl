variable "REPOSITORY" {
  default = "quay.io/vllm/automation-vllm"
}

variable "RELEASE_IMAGE" {
  default = false
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
  default = "0.7.1"
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
    VLLM_TGIS_ADAPTER_VERSION = "${VLLM_TGIS_ADAPTER_VERSION}"
    CUDA_MAJOR =  "12"
    CUDA_MINOR =  "8"
  }

  tags = [
    "${REPOSITORY}:${replace(VLLM_VERSION, "+", "_")}", # vllm_version might contain local version specifiers (+) which are not valid tags
    "${REPOSITORY}:cuda-${GITHUB_SHA}",
    "${REPOSITORY}:cuda-${GITHUB_RUN_ID}",
    RELEASE_IMAGE ? "quay.io/vllm/vllm-cuda:${replace(VLLM_VERSION, "+", "_")}" : ""
  ]
}

target "rocm" {
  inherits = ["_common"]
  dockerfile = "Dockerfile.rocm.ubi"

  args = {
    PYTHON_VERSION = "${PYTHON_VERSION}"
    ROCM_VERSION = "${ROCM_VERSION}"
    VLLM_TGIS_ADAPTER_VERSION = "${VLLM_TGIS_ADAPTER_VERSION}"
  }

  tags = [
    "${REPOSITORY}:${replace(VLLM_VERSION, "+", "_")}", # vllm_version might contain local version specifiers (+) which are not valid tags
    "${REPOSITORY}:rocm-${GITHUB_SHA}",
    "${REPOSITORY}:rocm-${GITHUB_RUN_ID}",
    RELEASE_IMAGE ? "quay.io/vllm/vllm-rocm:${replace(VLLM_VERSION, "+", "_")}" : ""
  ]
}
