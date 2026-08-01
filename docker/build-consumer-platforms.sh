#!/usr/bin/env bash
# Build the consumer-NVIDIA-platform image (sm_86 + sm_121) on the appmana
# cluster's buildkitd and push it to GHCR.
#
# Why this builder: arm64 is emulated via appmana's binfmt DaemonSet. The
# alternatives fail: GitHub's hosted arm64 runners OOM under nvcc, hilton's
# buildkitd has no CNI (no RUN egress), and a hand build on a serving Spark
# exhausts the unified-memory pool and wedges the node.
#
# Usage:
#   docker/build-consumer-platforms.sh [--platform linux/arm64] [--tag NAME]
#
# Requires: kubectl context `remote` (appmana-cluster-03), gh auth, buildctl.

set -euo pipefail

PLATFORM="${PLATFORM:-linux/arm64}"
IMAGE="${IMAGE:-ghcr.io/appmana/vllm-consumer:sm86-sm121}"
REPO_URL="${REPO_URL:-https://github.com/AppMana/forks-vllm-consumer-nvidia-platforms.git}"
REF="${REF:-appmana/vllm-consumer-nvidia-platforms}"
CONTEXT_NS="${CONTEXT_NS:-buildkit}"
KUBE_CONTEXT="${KUBE_CONTEXT:-remote}"
LOCAL_PORT="${LOCAL_PORT:-11234}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile}"
TARGET="${TARGET:-vllm-openai}"
BASE_IMAGE="${BASE_IMAGE:-}"

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

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"; [ -n "${pf_pid:-}" ] && kill "$pf_pid" 2>/dev/null || true' EXIT
chmod 700 "$workdir"

resolved_commit="$(git ls-remote "$REPO_URL" "refs/heads/$REF" | awk 'NR == 1 { print $1 }')"
if [ -z "$resolved_commit" ]; then
    resolved_commit="$(git ls-remote "$REPO_URL" "refs/tags/$REF^{}" "refs/tags/$REF" | awk 'NR == 1 { print $1 }')"
fi
if ! [[ "$resolved_commit" =~ ^[0-9a-f]{40}$ ]]; then
    echo "could not resolve $REF to an exact commit" >&2
    exit 1
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

kubectl --context "$KUBE_CONTEXT" port-forward -n "$CONTEXT_NS" svc/buildkitd \
    "$LOCAL_PORT:1234" >/dev/null 2>&1 &
pf_pid=$!
sleep 5

echo "building $IMAGE for $PLATFORM from $REF at $resolved_commit"

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
exec buildctl \
    --addr "tcp://127.0.0.1:$LOCAL_PORT" \
    "${tls_options[@]}" \
    build \
    --frontend dockerfile.v0 \
    "${build_options[@]}" \
    --secret "id=GIT_AUTH_TOKEN,src=$workdir/ghtoken" \
    --output "type=image,name=$IMAGE,push=true" \
    --progress plain
