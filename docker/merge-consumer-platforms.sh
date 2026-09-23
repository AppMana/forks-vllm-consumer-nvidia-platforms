#!/usr/bin/env bash
# Combine per-platform images from build-consumer-platforms.sh into one
# manifest list and check that both platforms are in it.
#
# Usage:
#   docker/merge-consumer-platforms.sh <target tag> <amd64 image> <arm64 image>
#
# Pass digests (name@sha256:...) for the sources: a tag can move between the
# build and the merge.

set -euo pipefail

[ $# -eq 3 ] || { echo "usage: $0 <target> <amd64 image> <arm64 image>" >&2; exit 2; }
target="$1" amd64="$2" arm64="$3"

docker buildx imagetools create --tag "$target" "$amd64" "$arm64"

platforms="$(docker buildx imagetools inspect --raw "$target" | python3 -c '
import json, sys
index = json.load(sys.stdin)
print(" ".join(sorted(
    m["platform"]["os"] + "/" + m["platform"]["architecture"]
    for m in index.get("manifests", [])
    if m.get("platform", {}).get("os") != "unknown"
)))
')"
if [ "$platforms" != "linux/amd64 linux/arm64" ]; then
    echo "$target has platforms [$platforms], expected [linux/amd64 linux/arm64]" >&2
    exit 1
fi
docker buildx imagetools inspect "$target"
