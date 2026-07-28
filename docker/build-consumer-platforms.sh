#!/usr/bin/env bash
# Build the consumer-NVIDIA-platform image (sm_86 + sm_121) on the appmana
# cluster's buildkitd and push it to GHCR.
#
# Why this builder: the arm64 half is emulated through the binfmt DaemonSet on
# appmana (src/clusters/appmana-cluster-03/buildkit/binfmt.yaml), so amd64
# workers can emit arm64 layers. Native alternatives do not work -- GitHub's
# hosted arm64 runners are 4 cores / 16 GB and are OOM-killed by nvcc, and the
# hilton buildkitd's containerd worker has no CNI, so RUN steps have no network
# egress. Never build on a Spark by hand: a ~30 GB build + image import on a
# serving node exhausts the unified-memory pool and wedges it.
#
# Usage:
#   docker/build-consumer-platforms.sh [--platform linux/arm64] [--tag NAME]
#
# Requires: kubectl context `remote` (appmana-cluster-03), gh auth, buildctl.

set -euo pipefail

PLATFORM="${PLATFORM:-linux/arm64}"
IMAGE="${IMAGE:-ghcr.io/appmana/vllm-consumer:sm86-sm121}"
REPO_URL="${REPO_URL:-https://github.com/AppMana/forks-vllm-ampere.git}"
REF="${REF:-appmana/vllm-consumer-nvidia-platforms}"
CONTEXT_NS="${CONTEXT_NS:-buildkit}"
KUBE_CONTEXT="${KUBE_CONTEXT:-remote}"
LOCAL_PORT="${LOCAL_PORT:-11234}"

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

# buildkitd requires mTLS; the client keypair lives in the cluster.
kubectl --context "$KUBE_CONTEXT" get secret buildkit-client-tls -n "$CONTEXT_NS" -o json \
  | python3 -c "
import base64, json, pathlib, sys
data = json.load(sys.stdin)['data']
out = pathlib.Path('$workdir')
for name, value in data.items():
    path = out / name
    path.write_bytes(base64.b64decode(value))
    path.chmod(0o600)
"

# GIT_AUTH_TOKEN authenticates the private git context. The token must not carry
# a trailing newline: `gh auth token` emits one and it corrupts the auth header,
# which surfaces as "could not read Username for 'https://github.com'".
gh auth token | tr -d '\n' > "$workdir/ghtoken"
chmod 600 "$workdir/ghtoken"

kubectl --context "$KUBE_CONTEXT" port-forward -n "$CONTEXT_NS" svc/buildkitd \
    "$LOCAL_PORT:1234" >/dev/null 2>&1 &
pf_pid=$!
sleep 5

echo "building $IMAGE for $PLATFORM from $REF"

# BUILDKIT_CONTEXT_KEEP_GIT_DIR: a git context strips .git by default, but the
# Dockerfile bind-mounts it for tools/check_repo.sh, which otherwise fails with
# 'failed to calculate checksum of ref ...: "/.git": not found'.
exec buildctl \
    --addr "tcp://127.0.0.1:$LOCAL_PORT" \
    --tlscacert "$workdir/ca.crt" \
    --tlscert "$workdir/tls.crt" \
    --tlskey "$workdir/tls.key" \
    --tlsservername buildkitd \
    build \
    --frontend dockerfile.v0 \
    --opt "context=${REPO_URL}#${REF}" \
    --opt filename=docker/Dockerfile \
    --opt target=vllm-openai \
    --opt "platform=$PLATFORM" \
    --opt build-arg:BUILDKIT_CONTEXT_KEEP_GIT_DIR=1 \
    --opt "build-arg:max_jobs=$MAX_JOBS" \
    --opt "build-arg:nvcc_threads=$NVCC_THREADS" \
    --opt build-arg:RUN_WHEEL_CHECK=false \
    --secret "id=GIT_AUTH_TOKEN,src=$workdir/ghtoken" \
    --output "type=image,name=$IMAGE,push=true" \
    --progress plain
