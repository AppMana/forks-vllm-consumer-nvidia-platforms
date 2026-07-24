# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-spark NCCL all-reduce over the dual-rail RoCE link (GB10).

Validates the whole TP=2 transport premise before any model runs: NCCL must
span BOTH per-PCIe-domain HCAs of the single 200G ConnectX-7 cable
(NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1) to exceed one domain's ~100G, and
GPUDirect on GB10's unified memory should ride dma-buf (no nvidia_peermem
module) — verified from the NCCL_DEBUG=INFO transport lines.

Run (spark-2ab3 is rank 0 / master 10.255.0.1, spark-5867 rank 1):

  NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 \
  NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_DEBUG=INFO \
  venv/bin/python nccl_dual_rail_bench.py --rank {0,1}

Compare a single-rail run (NCCL_IB_HCA=rocep1s0f1) to see the dual-rail win.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, choices=(0, 1))
    parser.add_argument("--master", default="10.255.0.1")
    parser.add_argument("--port", type=int, default=29411)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    os.environ.setdefault("MASTER_ADDR", args.master)
    os.environ.setdefault("MASTER_PORT", str(args.port))
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=args.rank, world_size=2)

    if args.rank == 0:
        print(f"# NCCL_IB_HCA={os.environ.get('NCCL_IB_HCA', '<unset>')}")
        print(f"{'size':>10} {'time_ms':>9} {'algbw_GB/s':>10} {'busbw_Gb/s':>10}")

    for size_mb in (1, 4, 16, 64, 256, 1024):
        n = size_mb * 1024 * 1024 // 2  # bf16 elements
        x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
        # warmup
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        for _ in range(args.iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / args.iters
        nbytes = n * 2
        # ring all-reduce bus bandwidth: 2*(n-1)/n * bytes / time; n=2 ranks
        algbw = nbytes / elapsed / 1e9
        busbw = algbw * (2 * (2 - 1) / 2)
        if args.rank == 0:
            print(
                f"{size_mb:>8}MB {elapsed * 1e3:>9.2f} {algbw:>10.2f} "
                f"{busbw * 8:>10.1f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
