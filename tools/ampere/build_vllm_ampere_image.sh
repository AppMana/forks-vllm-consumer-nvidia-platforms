#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

image_repo="${IMAGE_REPO:-harbor.appmana.com/appmana/vllm-ampere}"
commit="${COMMIT:-$(git -C "${repo_root}" rev-parse --short=9 HEAD)}"
build_commit="${VLLM_BUILD_COMMIT:-$(git -C "${repo_root}" rev-parse HEAD)}"
tag="${TAG:-${image_repo}:${commit}}"
cache_ref="${CACHE_REF:-${image_repo}:buildcache}"
builder="${BUILDER:-buildkit-linux}"
cache_from="${BUILDKIT_CACHE_FROM:-${cache_ref}}"
cache_to="${BUILDKIT_CACHE_TO:-${cache_ref}}"
cache_import="${BUILDKIT_CACHE_IMPORT:-1}"
cache_export="${BUILDKIT_CACHE_EXPORT:-1}"
cache_export_mode="${BUILDKIT_CACHE_EXPORT_MODE:-max}"

if ! docker buildx inspect "${builder}" >/dev/null 2>&1; then
  if [[ "${builder}" == "buildkit-linux" ]]; then
    echo "Buildx builder 'buildkit-linux' is not configured." >&2
    echo "Run from the appmana repo root:" >&2
    echo "  mkdir -p \$HOME/.buildkit-certs" >&2
    echo "  kubectl -n appmana get secret buildkit-client-tls -o jsonpath='{.data.ca\\.crt}' | base64 -d > \$HOME/.buildkit-certs/ca.pem" >&2
    echo "  kubectl -n appmana get secret buildkit-client-tls -o jsonpath='{.data.tls\\.crt}' | base64 -d > \$HOME/.buildkit-certs/cert.pem" >&2
    echo "  kubectl -n appmana get secret buildkit-client-tls -o jsonpath='{.data.tls\\.key}' | base64 -d > \$HOME/.buildkit-certs/key.pem" >&2
    echo "  chmod 600 \$HOME/.buildkit-certs/*.pem" >&2
    echo "  docker buildx create --name buildkit-linux --driver remote --driver-opt cacert=\$HOME/.buildkit-certs/ca.pem,cert=\$HOME/.buildkit-certs/cert.pem,key=\$HOME/.buildkit-certs/key.pem,servername=buildkitd.buildkit.svc.cluster.local tcp://10.152.184.74:1234" >&2
  else
    echo "Buildx builder '${builder}' is not configured." >&2
  fi
  exit 1
fi

max_jobs="${MAX_JOBS:-8}"
nvcc_threads="${NVCC_THREADS:-4}"
torch_arch_list="${TORCH_CUDA_ARCH_LIST:-8.6}"
flashinfer_download_cubin="${FLASHINFER_DOWNLOAD_CUBIN:-0}"
use_sccache="${USE_SCCACHE:-0}"
sccache_endpoint="${SCCACHE_ENDPOINT:-http://10.152.184.210:8333}"
sccache_bucket="${SCCACHE_BUCKET_NAME:-appmana-private}"
sccache_region="${SCCACHE_REGION_NAME:-us-west-2}"
sccache_s3_no_credentials="${SCCACHE_S3_NO_CREDENTIALS:-0}"
sccache_recache="${SCCACHE_RECACHE:-0}"
skip_flash_attn_build="${VLLM_SKIP_FLASH_ATTN_BUILD:-0}"
install_kv_connectors="${INSTALL_KV_CONNECTORS:-true}"
lmcache_git_ref="${LMCACHE_GIT_REF:-cd51f3a15766e326f92998c072264a5a6caa4efe}"
appmana_nccl_git_repo="${APPMANA_NCCL_GIT_REPO:-https://github.com/AppMana/forks-nccl-rdma-routing.git}"
appmana_nccl_git_ref="${APPMANA_NCCL_GIT_REF:-b8fabdc5145fab760c1a9acce2892ac6077b1679}"
vllm_source_repo="${VLLM_SOURCE_REPO:-https://github.com/AppMana/forks-vllm-ampere}"

secret_args=()
cache_args=()
aws_credentials_file=""

if [[ "${use_sccache}" == "1" ]]; then
  if [[ -n "${SCCACHE_AWS_CREDENTIALS_FILE:-}" ]]; then
    aws_credentials_file="${SCCACHE_AWS_CREDENTIALS_FILE}"
  else
    access_key="${SCCACHE_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}"
    secret_key="${SCCACHE_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"

    if [[ -z "${access_key}" || -z "${secret_key}" ]]; then
      if command -v kubectl >/dev/null 2>&1; then
        access_key="$(kubectl -n buildkit get secret seaweedfs-s3 -o jsonpath='{.data.AWS_ACCESS_KEY_ID}' 2>/dev/null | base64 -d || true)"
        secret_key="$(kubectl -n buildkit get secret seaweedfs-s3 -o jsonpath='{.data.AWS_SECRET_ACCESS_KEY}' 2>/dev/null | base64 -d || true)"
      fi
    fi

    if [[ -z "${access_key}" || -z "${secret_key}" ]]; then
      echo "USE_SCCACHE=1 but no S3 credentials were available." >&2
      echo "Set SCCACHE_AWS_ACCESS_KEY_ID/SCCACHE_AWS_SECRET_ACCESS_KEY, SCCACHE_AWS_CREDENTIALS_FILE, or apply the buildkit/seaweedfs-s3 Secret." >&2
      exit 1
    fi

    aws_credentials_file="$(mktemp)"
    trap 'rm -f "${aws_credentials_file}"' EXIT
    cat >"${aws_credentials_file}" <<EOF
[default]
aws_access_key_id=${access_key}
aws_secret_access_key=${secret_key}
EOF
  fi
  secret_args+=(--secret "id=aws-credentials,src=${aws_credentials_file}")
fi

if [[ "${cache_import}" != "0" && -n "${cache_from}" ]]; then
  cache_args+=(--cache-from "type=registry,ref=${cache_from}")
fi

if [[ "${cache_export}" != "0" && -n "${cache_to}" ]]; then
  cache_args+=(--cache-to "type=registry,ref=${cache_to},mode=${cache_export_mode}")
fi

docker buildx build "${repo_root}" \
  --builder "${builder}" \
  --file "${repo_root}/docker/Dockerfile" \
  --target vllm-openai \
  --tag "${tag}" \
  --build-arg "max_jobs=${max_jobs}" \
  --build-arg "nvcc_threads=${nvcc_threads}" \
  --build-arg "torch_cuda_arch_list=${torch_arch_list}" \
  --build-arg "FLASHINFER_DOWNLOAD_CUBIN=${flashinfer_download_cubin}" \
  --build-arg "USE_SCCACHE=${use_sccache}" \
  --build-arg "SCCACHE_ENDPOINT=${sccache_endpoint}" \
  --build-arg "SCCACHE_BUCKET_NAME=${sccache_bucket}" \
  --build-arg "SCCACHE_REGION_NAME=${sccache_region}" \
  --build-arg "SCCACHE_S3_NO_CREDENTIALS=${sccache_s3_no_credentials}" \
  --build-arg "SCCACHE_RECACHE=${sccache_recache}" \
  --build-arg "VLLM_SKIP_FLASH_ATTN_BUILD=${skip_flash_attn_build}" \
  --build-arg "INSTALL_KV_CONNECTORS=${install_kv_connectors}" \
  --build-arg "LMCACHE_GIT_REF=${lmcache_git_ref}" \
  --build-arg "APPMANA_NCCL_GIT_REPO=${appmana_nccl_git_repo}" \
  --build-arg "APPMANA_NCCL_GIT_REF=${appmana_nccl_git_ref}" \
  --build-arg "VLLM_BUILD_COMMIT=${build_commit}" \
  --build-arg "VLLM_SOURCE_REPO=${vllm_source_repo}" \
  --build-arg "VLLM_IMAGE_TAG=${tag}" \
  "${secret_args[@]}" \
  "${cache_args[@]}" \
  "$@"
