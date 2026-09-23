#!/usr/bin/env bash
# Build the consumer-NVIDIA-platform image (sm_86 + sm_121) on the appmana
# cluster's buildkitd and push it to GHCR.
#
# Default builder: appmana's buildkitd-vllm, with arm64 emulated via its binfmt
# DaemonSet. Native arm64 alternative: hilton's buildkitd DaemonSet on the
# Sparks (RUN steps use the host network), only while no model is serving
# there; a build on a serving Spark exhausts the unified-memory pool. GitHub's
# hosted arm64 runners OOM under nvcc.
#
#   KUBE_CONTEXT=Default KUBECONFIG=<hilton kubeconfig> \
#   BUILDKIT_TARGET=pod/<buildkitd pod on the idle Spark> USE_SCCACHE=0 \
#   CACHE_REF=ghcr.io/appmana/vllm-consumer:buildcache-arm64 \
#     docker/build-consumer-platforms.sh --platform linux/arm64 --tag ...
#
# USE_SCCACHE=0 there because hilton cannot reach appmana's S3; the GHCR layer
# cache still applies. Combine per-platform tags with
# docker/merge-consumer-platforms.sh.
#
# Usage:
#   docker/build-consumer-platforms.sh [--platform linux/arm64] [--tag NAME]
#       [--ref BRANCH|TAG|COMMIT]
#
# The default tag is sm86-sm121-<commit10>-<arch>; tags are never reused for
# different content (MERGING.md, release gate).
#
# Caches: the build dials one fixed buildkitd pod (Service buildkitd-vllm) so
# its local layer cache persists between runs, imports and exports the layer
# cache at $CACHE_REF in GHCR, and compiles through sccache against the
# cluster's S3 (credentials from the buildkit/seaweedfs-s3 Secret). Leave all
# three on; a build without them recompiles vLLM for over an hour.
#
# Requires: kubectl context `local` (appmana-cluster-03), gh auth, buildctl.

set -euo pipefail

PLATFORM="${PLATFORM:-linux/arm64}"
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/appmana/vllm-consumer}"
IMAGE="${IMAGE:-}"
REPO_URL="${REPO_URL:-https://github.com/AppMana/forks-vllm-consumer-nvidia-platforms.git}"
REF="${REF:-appmana/vllm-consumer-nvidia-platforms}"
CONTEXT_NS="${CONTEXT_NS:-buildkit}"
KUBE_CONTEXT="${KUBE_CONTEXT:-local}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile}"
# An explicitly empty TARGET builds the Dockerfile's final stage. This is used
# by single-stage overlay Dockerfiles.
TARGET="${TARGET-vllm-openai}"
BASE_IMAGE="${BASE_IMAGE:-}"
BUILDKIT_SERVICE="${BUILDKIT_SERVICE:-buildkitd-vllm}"
# What to port-forward to. A pod name pins the build to one node, which a
# Service cannot do through kubectl port-forward.
BUILDKIT_TARGET="${BUILDKIT_TARGET:-svc/$BUILDKIT_SERVICE}"
# Per-architecture layer cache: two mode=max exports to one ref overwrite each
# other. amd64 keeps the historical `buildcache` ref, which is the warm one.
CACHE_REF="${CACHE_REF:-}"
USE_SCCACHE="${USE_SCCACHE:-1}"
# LMCache, NIXL and Mooncake, as in the Harbor sm86 image; every one of them
# ships aarch64 wheels, so both platforms carry the same connectors.
INSTALL_KV_CONNECTORS="${INSTALL_KV_CONNECTORS:-true}"
SCCACHE_ENDPOINT="${SCCACHE_ENDPOINT:-http://10.152.184.210:8333}"
SCCACHE_BUCKET_NAME="${SCCACHE_BUCKET_NAME:-appmana-private}"
SCCACHE_REGION_NAME="${SCCACHE_REGION_NAME:-us-west-2}"

# nvcc parallelism. The arm64 stages run under QEMU, so wall-clock is already
# poor; oversubscribing turns it into OOM. Raise only with headroom measured on
# the worker.
MAX_JOBS="${MAX_JOBS:-8}"
NVCC_THREADS="${NVCC_THREADS:-2}"

while [ $# -gt 0 ]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --tag)      IMAGE="$2"; shift 2 ;;
        --ref)      REF="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

command -v buildctl >/dev/null || { echo "buildctl not on PATH" >&2; exit 1; }

case "$PLATFORM" in
    linux/amd64) arch=amd64 ;;
    linux/arm64) arch=arm64 ;;
    *) echo "unsupported platform: $PLATFORM" >&2; exit 2 ;;
esac

# The workdir holds the GitHub token and S3 credentials, and the port-forward
# must not outlive the build, so buildctl runs as a child (not exec) and this
# trap always fires.
workdir="$(mktemp -d)"
cleanup() {
    if [ -n "${pf_pid:-}" ]; then
        kill "$pf_pid" 2>/dev/null || true
        wait "$pf_pid" 2>/dev/null || true
    fi
    rm -rf "$workdir"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
chmod 700 "$workdir"

if [[ "$REF" =~ ^[0-9a-f]{40}$ ]]; then
    resolved_commit="$REF"
else
    resolved_commit="$(git ls-remote "$REPO_URL" "refs/heads/$REF" | awk 'NR == 1 { print $1 }')"
    if [ -z "$resolved_commit" ]; then
        resolved_commit="$(git ls-remote "$REPO_URL" "refs/tags/$REF^{}" "refs/tags/$REF" | awk 'NR == 1 { print $1 }')"
    fi
fi
if ! [[ "$resolved_commit" =~ ^[0-9a-f]{40}$ ]]; then
    echo "could not resolve $REF to an exact commit (branch, tag or full 40-hex SHA)" >&2
    exit 1
fi
IMAGE="${IMAGE:-$IMAGE_REPO:sm86-sm121-${resolved_commit:0:10}-$arch}"
if [ -z "$CACHE_REF" ]; then
    if [ "$arch" = amd64 ]; then
        CACHE_REF="$IMAGE_REPO:buildcache"
    else
        CACHE_REF="$IMAGE_REPO:buildcache-$arch"
    fi
fi
wheel_version="${VLLM_VERSION_OVERRIDE:-0.0.0+consumer.${resolved_commit:0:10}}"

# Use mTLS when the BuildKit deployment publishes a client certificate. A
# cluster-local BuildKit can instead rely on the authenticated Kubernetes
# port-forward and does not need a second secret.
tls_options=()
if kubectl --context "$KUBE_CONTEXT" get secret buildkit-client-tls \
    -n "$CONTEXT_NS" -o json > "$workdir/buildkit-client-tls.json" 2>/dev/null; then
  python3 -c "
import base64, json, pathlib, sys
data = json.load(open('$workdir/buildkit-client-tls.json'))['data']
out = pathlib.Path('$workdir')
for name, value in data.items():
    path = out / name
    path.write_bytes(base64.b64decode(value))
    path.chmod(0o600)
"
  tls_options=(
      --tlscacert "$workdir/ca.crt"
      --tlscert "$workdir/tls.crt"
      --tlskey "$workdir/tls.key"
      --tlsservername buildkitd
  )
fi

# GIT_AUTH_TOKEN authenticates the private git context. The token must not carry
# a trailing newline: `gh auth token` emits one and it corrupts the auth header,
# which surfaces as "could not read Username for 'https://github.com'".
gh auth token | tr -d '\n' > "$workdir/ghtoken"
chmod 600 "$workdir/ghtoken"

secret_options=(--secret "id=GIT_AUTH_TOKEN,src=$workdir/ghtoken")
sccache_options=()
if [ "$USE_SCCACHE" = "1" ]; then
    access_key="$(kubectl --context "$KUBE_CONTEXT" -n "$CONTEXT_NS" get secret seaweedfs-s3 -o jsonpath='{.data.AWS_ACCESS_KEY_ID}' | base64 -d)"
    secret_key="$(kubectl --context "$KUBE_CONTEXT" -n "$CONTEXT_NS" get secret seaweedfs-s3 -o jsonpath='{.data.AWS_SECRET_ACCESS_KEY}' | base64 -d)"
    if [ -z "$access_key" ] || [ -z "$secret_key" ]; then
        echo "USE_SCCACHE=1 but the $CONTEXT_NS/seaweedfs-s3 Secret has no S3 credentials" >&2
        exit 1
    fi
    printf '[default]\naws_access_key_id=%s\naws_secret_access_key=%s\n' "$access_key" "$secret_key" > "$workdir/aws-credentials"
    chmod 600 "$workdir/aws-credentials"
    secret_options+=(--secret "id=aws-credentials,src=$workdir/aws-credentials")
    sccache_options=(
        --opt build-arg:USE_SCCACHE=1
        --opt "build-arg:SCCACHE_ENDPOINT=$SCCACHE_ENDPOINT"
        --opt "build-arg:SCCACHE_BUCKET_NAME=$SCCACHE_BUCKET_NAME"
        --opt "build-arg:SCCACHE_REGION_NAME=$SCCACHE_REGION_NAME"
    )
fi
# Let kubectl pick a free local port and read it back, so a stale forward left
# on a fixed port by another run can never be reused silently.
kubectl --context "$KUBE_CONTEXT" port-forward -n "$CONTEXT_NS" "$BUILDKIT_TARGET" \
    ":1234" > "$workdir/port-forward.log" 2>&1 &
pf_pid=$!
local_port=""
for _ in $(seq 1 30); do
    if ! kill -0 "$pf_pid" 2>/dev/null; then
        echo "port-forward to $BUILDKIT_TARGET exited:" >&2
        cat "$workdir/port-forward.log" >&2
        exit 1
    fi
    local_port="$(sed -n 's/^Forwarding from 127\.0\.0\.1:\([0-9]*\) -> 1234$/\1/p' "$workdir/port-forward.log" | head -n 1)"
    [ -n "$local_port" ] && break
    sleep 1
done
if [ -z "$local_port" ]; then
    echo "port-forward to $BUILDKIT_TARGET did not come up:" >&2
    cat "$workdir/port-forward.log" >&2
    exit 1
fi
buildctl --addr "tcp://127.0.0.1:$local_port" "${tls_options[@]}" debug workers >/dev/null

echo "building $IMAGE for $PLATFORM from $REF at $resolved_commit (cache $CACHE_REF)"

build_options=(
    --opt "context=${REPO_URL}#${resolved_commit}"
    --opt "filename=$DOCKERFILE"
    --opt "platform=$PLATFORM"
    --opt build-arg:BUILDKIT_CONTEXT_KEEP_GIT_DIR=1
    --opt "build-arg:max_jobs=$MAX_JOBS"
    --opt "build-arg:nvcc_threads=$NVCC_THREADS"
    --opt "build-arg:VLLM_BUILD_COMMIT=$resolved_commit"
    --opt "build-arg:VLLM_IMAGE_TAG=$IMAGE"
    --opt "build-arg:VLLM_VERSION_OVERRIDE=$wheel_version"
    --opt build-arg:RUN_WHEEL_CHECK=false
    --opt "build-arg:INSTALL_KV_CONNECTORS=$INSTALL_KV_CONNECTORS"
)
if [ -n "$TARGET" ]; then
    build_options+=(--opt "target=$TARGET")
fi
if [ -n "$BASE_IMAGE" ]; then
    build_options+=(--opt "build-arg:BASE_IMAGE=$BASE_IMAGE")
fi

# BUILDKIT_CONTEXT_KEEP_GIT_DIR: a git context strips .git by default, but the
# Dockerfile bind-mounts it for tools/check_repo.sh, which otherwise fails with
# 'failed to calculate checksum of ref ...: "/.git": not found'.
buildctl \
    --addr "tcp://127.0.0.1:$local_port" \
    "${tls_options[@]}" \
    build \
    --frontend dockerfile.v0 \
    "${build_options[@]}" \
    "${sccache_options[@]}" \
    "${secret_options[@]}" \
    --import-cache "type=registry,ref=$CACHE_REF" \
    --export-cache "type=registry,ref=$CACHE_REF,mode=max" \
    --output "type=image,name=$IMAGE,push=true" \
    --progress plain
