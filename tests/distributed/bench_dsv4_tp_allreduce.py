"""Measure the TP collective used by an M-token DSV4 hidden-state shard."""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29601)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["LOCAL_RANK"] = "0"
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")

    if args.rank == 0:
        print(
            "NCCL "
            + " ".join(
                f"{name}={os.environ.get(name, '')}"
                for name in (
                    "NCCL_IB_HCA",
                    "NCCL_IB_QPS_PER_CONNECTION",
                    "NCCL_IB_SPLIT_DATA_ON_QPS",
                    "NCCL_SOCKET_IFNAME",
                    "NCCL_IB_DISABLE",
                    "NCCL_NET",
                )
            ),
            flush=True,
        )

    for tokens in (1024, 8184):
        tensor = torch.ones((tokens, 4096), dtype=torch.bfloat16, device="cuda")
        for _ in range(5):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        for _ in range(args.iterations):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if args.rank == 0:
            payload_bytes = tensor.numel() * tensor.element_size()
            latency_ms = elapsed * 1000 / args.iterations
            algorithm_gbps = payload_bytes / (latency_ms / 1000) * 8 / 1e9
            bus_gbps = algorithm_gbps * 2 * (args.world_size - 1) / args.world_size
            print(
                f"tokens={tokens} bytes={payload_bytes} latency_ms={latency_ms:.3f} "
                f"algbw_gbps={algorithm_gbps:.2f} busbw_gbps={bus_gbps:.2f}",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
