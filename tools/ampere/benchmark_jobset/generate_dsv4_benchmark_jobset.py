#!/usr/bin/env python3
"""Generate an applied-only DeepSeek V4 benchmark JobSet.

The default expanded form creates one one-pod replicatedJob per rank so a rank
can mount the matching LeaderWorkerSet local-path PVC. The ``--indexed`` form
keeps a single indexed replicatedJob, which is the shape currently admitted by
Kueue on the tb-chain queue; because Kubernetes cannot template PVC claim names
from completion indexes, indexed jobs use the template's shared/ephemeral cache
layout instead of per-rank LWS PVCs.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path
from typing import Any

import yaml


def verify_image_exists(image: str) -> None:
    """Fail before a 10-node slice sits in ImagePullBackOff.

    A mistyped tag costs a full Kueue admission cycle to discover, and the
    kubelet reports it as a pull failure rather than a bad reference.
    """
    try:
        proc = subprocess.run(
            ["docker", "manifest", "inspect", image],
            capture_output=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(
            f"cannot verify {image}: {exc}. Pass --no-verify-image to skip."
        ) from exc
    if proc.returncode != 0:
        raise SystemExit(
            f"image not found in registry: {image}\n"
            f"  {proc.stderr.decode().strip()}\n"
            "Check the tag (the build script uses `git rev-parse --short=9`)."
        )


# Full 12-node chain in tb-chain-index order (idx 0..11). The boot script
# intersects this order with the nodes Kueue TAS actually places on, so listing
# the WHOLE chain (not a fixed 10-window) makes the benchmark robust to whichever
# contiguous world-size run TAS selects. Keep in index order == physical PP order.
RANK_NODES = [
    ("appmana-002", "10.2.0.41"),   # idx 0
    ("appmana-018", "10.2.0.56"),   # idx 1
    ("appmana-027", "10.2.0.4"),    # idx 2
    ("appmana-019", "10.2.0.57"),   # idx 3
    ("appmana-008", "10.2.0.50"),   # idx 4
    ("appmana-020", "10.2.0.58"),   # idx 5
    ("appmana-009", "10.2.0.15"),   # idx 6
    ("appmana-025", "10.2.0.67"),   # idx 7
    ("appmana-023", "10.2.0.61"),   # idx 8
    ("appmana-022", "10.2.0.60"),   # idx 9
    ("appmana-021", "10.2.0.59"),   # idx 10
    ("appmana-004", "10.2.0.9"),    # idx 11
]


def _parse_rank_nodes(value: str | None) -> list[tuple[str, str]]:
    if not value:
        return RANK_NODES
    nodes: list[tuple[str, str]] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "--rank-nodes entries must be HOST=IP, got " f"{item!r}"
            )
        host, ip = item.split("=", 1)
        if not host or not ip:
            raise ValueError(
                "--rank-nodes entries must be HOST=IP, got " f"{item!r}"
            )
        nodes.append((host, ip))
    if not nodes:
        raise ValueError("--rank-nodes must contain at least one HOST=IP entry")
    return nodes


def _set_env(env: list[dict[str, Any]], name: str, value: str) -> None:
    for item in env:
        if item.get("name") == name:
            item["value"] = value
            return
    env.append({"name": name, "value": value})


CONTROLLER_OWNED_NCCL_ENV = {
    "GLOO_SOCKET_IFNAME",
    "NCCL_ALGO",
    "NCCL_IB_ADDR_FAMILY",
    "NCCL_IB_HCA",
    "NCCL_IB_MERGE_NICS",
    "NCCL_IB_SUBNET_AWARE_ROUTING",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_NET_MERGE_LEVEL",
    "NCCL_PROTO",
    "NCCL_SOCKET_IFNAME",
}


def _scrub_controller_owned_env(
    env: list[dict[str, Any]], env_overrides: dict[str, str]
) -> None:
    """Keep benchmark JobSets from bypassing the tb-chain webhook defaults."""
    override_names = set(env_overrides)
    env[:] = [
        item
        for item in env
        if item.get("name") not in CONTROLLER_OWNED_NCCL_ENV
        or item.get("name") in override_names
    ]


def _rank_pvc(rank: int) -> str:
    if rank == 0:
        return "jit-cache-local-vllm-deepseek-v4-0"
    return f"jit-cache-local-vllm-deepseek-v4-0-{rank}"


def _rank_pvc_for_node(hostname: str) -> str:
    for rank, (rank_hostname, _) in enumerate(RANK_NODES):
        if rank_hostname == hostname:
            return _rank_pvc(rank)
    raise ValueError(f"no known LWS local PVC for host {hostname!r}")


def _ensure_rdma_access(pod_spec: dict[str, Any], container: dict[str, Any]) -> None:
    volumes = pod_spec.setdefault("volumes", [])
    if not any(volume.get("name") == "tb-chain-infiniband" for volume in volumes):
        volumes.append(
            {
                "name": "tb-chain-infiniband",
                "hostPath": {"path": "/dev/infiniband", "type": "Directory"},
            }
        )

    mounts = container.setdefault("volumeMounts", [])
    if not any(mount.get("mountPath") == "/dev/infiniband" for mount in mounts):
        mounts.append(
            {
                "name": "tb-chain-infiniband",
                "mountPath": "/dev/infiniband",
            }
        )

    security_context = container.setdefault("securityContext", {})
    capabilities = security_context.setdefault("capabilities", {})
    add = capabilities.setdefault("add", [])
    for capability in ("IPC_LOCK", "SYS_RESOURCE"):
        if capability not in add:
            add.append(capability)
    security_context["privileged"] = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--world-size", type=int, default=10)
    parser.add_argument(
        "--rank-nodes",
        default=None,
        help=(
            "Comma-separated HOST=IP rank list. Defaults to the DeepSeek "
            "serving chain slice in this file."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-kueue", action="store_true")
    parser.add_argument(
        "--no-verify-image",
        action="store_true",
        help="Skip the registry existence check for --image.",
    )
    parser.add_argument(
        "--indexed",
        action="store_true",
        help=(
            "Keep one indexed replicatedJob instead of expanding to one "
            "replicatedJob per rank. This is the shape admitted by Kueue."
        ),
    )
    parser.add_argument(
        "--shared-cache-pvc",
        default=None,
        help=(
            "Mount this PVC at /jit-local for every rank instead of the "
            "rank-local LeaderWorkerSet PVCs."
        ),
    )
    parser.add_argument(
        "--use-shared-model-path",
        action="store_true",
        help=(
            "Link rank shards from the shared Hugging Face snapshot instead "
            "of copying them into /jit-local. This is useful when logical "
            "rank order changes independently of physical LWS PVC placement."
        ),
    )
    parser.add_argument(
        "--prewarm-only",
        action="store_true",
        help="Populate the rank-local LWS PVCs and exit before starting Ray/vLLM.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Set or override a container environment variable on every rank.",
    )
    args = parser.parse_args()
    if not args.no_verify_image:
        verify_image_exists(args.image)
    env_overrides: dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            raise ValueError(f"--env must be NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        if not name:
            raise ValueError(f"--env name must be non-empty, got {item!r}")
        env_overrides[name] = value
    rank_nodes = _parse_rank_nodes(args.rank_nodes)

    docs = list(yaml.safe_load_all(args.template.read_text()))
    jobset = next(doc for doc in docs if doc and doc.get("kind") == "JobSet")
    jobset["metadata"]["name"] = args.name

    # Give every generated JobSet its own boot-script ConfigMap. The template's
    # shared name meant applying a second row silently rewrote the script the
    # first row was still running from (a Kueue-queued row picked up the next
    # row's flags mid-flight; observed on dsv4-m8-int8kv-fixed-003, which ran
    # with the async-scheduling flag belonging to dsv4-m7-async-on-002).
    script_cm = next(
        doc
        for doc in docs
        if doc and doc.get("kind") == "ConfigMap" and "script" in doc["metadata"]["name"]
    )
    old_script_name = script_cm["metadata"]["name"]
    new_script_name = f"{args.name}-script"
    script_cm["metadata"]["name"] = new_script_name
    for volume in jobset["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
        "spec"
    ].get("volumes", []):
        if volume.get("configMap", {}).get("name") == old_script_name:
            volume["configMap"]["name"] = new_script_name
    if args.no_kueue:
        jobset.get("metadata", {}).get("labels", {}).pop(
            "kueue.x-k8s.io/queue-name", None
        )

    rank_template = jobset["spec"]["replicatedJobs"][0]
    rank_jobs = []
    if args.world_size > len(rank_nodes):
        raise ValueError(
            f"world-size={args.world_size} exceeds configured ranks "
            f"({len(rank_nodes)})"
        )
    # Bake the FULL chain order (all configured ranks), not just the first
    # world_size. The boot-time rendezvous intersects this with the nodes Kueue
    # actually placed, so any contiguous world_size window TAS picks resolves to
    # exactly world_size IPs in chain order. Baking a fixed 10-window here breaks
    # whenever TAS selects a different contiguous run (e.g. idx 0-9 vs 2-11).
    chain_ips = ",".join(ip for _, ip in rank_nodes)
    if args.indexed:
        job = copy.deepcopy(rank_template)
        job["name"] = "rank"
        if not args.no_kueue:
            job.setdefault("template", {}).setdefault("metadata", {}).setdefault(
                "labels", {}
            )["kueue.x-k8s.io/queue-name"] = "tb-chain"
        spec = job["template"]["spec"]
        spec["completions"] = args.world_size
        spec["parallelism"] = args.world_size

        pod_spec = spec["template"]["spec"]
        pod_spec.setdefault("nodeSelector", {})["kubernetes.io/os"] = "linux"
        pod_spec.get("nodeSelector", {}).pop("kubernetes.io/hostname", None)
        pod_meta = spec["template"].setdefault("metadata", {})
        if args.no_kueue:
            for key in (
                "kueue.x-k8s.io/podset-required-topology",
                "kueue.x-k8s.io/podset-slice-required-topology",
                "kueue.x-k8s.io/podset-slice-size",
                "kueue.x-k8s.io/tas-ordered-evictable",
            ):
                pod_meta.get("annotations", {}).pop(key, None)
            pod_meta.get("labels", {}).pop("kueue.x-k8s.io/queue-name", None)
        else:
            pod_meta.setdefault("labels", {})[
                "kueue.x-k8s.io/queue-name"
            ] = "tb-chain"

        container = pod_spec["containers"][0]
        _ensure_rdma_access(pod_spec, container)
        container["image"] = args.image
        env = container.setdefault("env", [])
        _scrub_controller_owned_env(env, env_overrides)
        _set_env(env, "JOBSET_NAME", args.name)
        _set_env(env, "WORLD_SIZE", str(args.world_size))
        _set_env(env, "VLLM_RAY_WORKER_IP_ORDER", chain_ips)
        _set_env(
            env,
            "APPMANA_DSV4_USE_SHARED_MODEL_PATH",
            "1" if args.use_shared_model_path else "0",
        )
        _set_env(env, "APPMANA_DSV4_REQUIRE_LOCAL_MODEL_PATH", "0")
        _set_env(env, "APPMANA_DSV4_PREWARM_ONLY", "1" if args.prewarm_only else "0")
        _set_env(env, "NCCL_DEBUG", "WARN")
        for name, value in env_overrides.items():
            _set_env(env, name, value)

        # jit-cache-local holds the JIT compiler caches (Triton/inductor/cuda).
        # With --use-shared-model-path the model comes from the shared snapshot
        # via /tmp, so /jit-local is compiler-caches-ONLY and MUST be a node-local
        # emptyDir: Triton's os.replace() of every compiled .ttir throws EIO on a
        # SeaweedFS FUSE mount under the warmup write burst. Only bind it to the
        # shared PVC when the model actually lives on it (non-shared-model-path).
        if args.shared_cache_pvc and not args.use_shared_model_path:
            for volume in pod_spec["volumes"]:
                if volume.get("name") == "jit-cache-local":
                    volume.clear()
                    volume["name"] = "jit-cache-local"
                    volume["persistentVolumeClaim"] = {
                        "claimName": args.shared_cache_pvc
                    }
                    break
            else:
                raise RuntimeError("jit-cache-local volume not found")

        jobset["spec"]["replicatedJobs"] = [job]
        args.output.write_text("---\n" + yaml.safe_dump_all(docs, sort_keys=False))
        return

    for rank in range(args.world_size):
        job = copy.deepcopy(rank_template)
        job["name"] = f"rank-{rank}"
        if not args.no_kueue:
            job.setdefault("template", {}).setdefault("metadata", {}).setdefault(
                "labels", {}
            )["kueue.x-k8s.io/queue-name"] = "tb-chain"
        spec = job["template"]["spec"]
        spec["completions"] = 1
        spec["parallelism"] = 1
        spec.pop("completionMode", None)

        pod_spec = spec["template"]["spec"]
        pod_spec.setdefault("nodeSelector", {})["kubernetes.io/hostname"] = (
            rank_nodes[rank][0]
        )
        pod_meta = spec["template"].setdefault("metadata", {})
        if args.no_kueue:
            for key in (
                "kueue.x-k8s.io/podset-required-topology",
                "kueue.x-k8s.io/podset-slice-required-topology",
                "kueue.x-k8s.io/podset-slice-size",
                "kueue.x-k8s.io/tas-ordered-evictable",
            ):
                pod_meta.get("annotations", {}).pop(key, None)
            pod_meta.get("labels", {}).pop("kueue.x-k8s.io/queue-name", None)
        else:
            pod_meta.setdefault("labels", {})[
                "kueue.x-k8s.io/queue-name"
            ] = "tb-chain"
        container = pod_spec["containers"][0]
        _ensure_rdma_access(pod_spec, container)
        container["image"] = args.image
        env = container.setdefault("env", [])
        _scrub_controller_owned_env(env, env_overrides)
        _set_env(env, "JOBSET_NAME", args.name)
        _set_env(env, "WORLD_SIZE", str(args.world_size))
        _set_env(env, "VLLM_RAY_WORKER_IP_ORDER", chain_ips)
        _set_env(
            env,
            "APPMANA_DSV4_USE_SHARED_MODEL_PATH",
            "1" if args.use_shared_model_path else "0",
        )
        _set_env(
            env,
            "APPMANA_DSV4_REQUIRE_LOCAL_MODEL_PATH",
            "0",
        )
        _set_env(env, "APPMANA_DSV4_PREWARM_ONLY", "1" if args.prewarm_only else "0")
        _set_env(env, "NCCL_DEBUG", "WARN")
        for name, value in env_overrides.items():
            _set_env(env, name, value)

        for volume in pod_spec["volumes"]:
            if volume.get("name") == "jit-cache-local":
                volume.clear()
                volume["name"] = "jit-cache-local"
                if args.use_shared_model_path:
                    # Compiler caches only; model is on /tmp. Node-local emptyDir,
                    # never a network PVC (FUSE os.replace -> EIO during warmup).
                    volume["emptyDir"] = {"sizeLimit": "50Gi"}
                elif args.shared_cache_pvc:
                    volume["persistentVolumeClaim"] = {
                        "claimName": args.shared_cache_pvc
                    }
                else:
                    volume["persistentVolumeClaim"] = {
                        "claimName": _rank_pvc_for_node(rank_nodes[rank][0])
                    }
                break
        else:
            raise RuntimeError("jit-cache-local volume not found")

        rank_jobs.append(job)

    jobset["spec"]["replicatedJobs"] = rank_jobs
    args.output.write_text("---\n" + yaml.safe_dump_all(docs, sort_keys=False))


if __name__ == "__main__":
    main()
