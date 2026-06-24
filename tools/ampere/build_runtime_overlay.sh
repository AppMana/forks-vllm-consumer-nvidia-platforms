#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

base_image="${BASE_IMAGE:?set BASE_IMAGE to the already-built vLLM image, e.g. harbor.appmana.com/appmana/vllm-ampere:efc9daf72}"
image_repo="${IMAGE_REPO:-harbor.appmana.com/appmana/vllm-ampere}"
commit="${COMMIT:-$(git -C "${repo_root}" rev-parse --short=9 HEAD)}"
tag="${TAG:-${image_repo}:${commit}}"
builder="${BUILDER:-default}"

docker buildx build "${repo_root}" \
  --builder "${builder}" \
  --file "${repo_root}/docker/Dockerfile.ampere-runtime" \
  --tag "${tag}" \
  --build-arg "BASE_IMAGE=${base_image}" \
  "$@"
