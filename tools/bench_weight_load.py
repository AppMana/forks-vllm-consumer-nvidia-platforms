# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure where checkpoint load time goes inside vLLM's real loader path.

Subcommands:

``iterator``
    Drives ``vllm.model_executor.model_loader.weight_utils
    .safetensors_weights_iterator`` -- the exact generator
    ``DefaultModelLoader`` uses -- over a chosen number of real checkpoint
    shards, under a chosen ``--safetensors-load-strategy``.  Time is split
    between the generator's ``__next__`` (open/read/dtype/pin) and the
    consumer (optional host-to-device copy, which is what a model's
    ``weight_loader`` ultimately performs).

``raw``
    Cold sequential ``read()`` of the same shards with no parsing at all.
    This is the storage ceiling every other number is measured against.

``engine``
    Real ``vllm.LLM`` construction on a checkpoint small enough to fit the
    local GPUs, with timing wrappers installed around the real weight
    iterator and the real ``process_weights_after_loading`` so the phases can
    be attributed without substituting any of them.

``flashpack``
    Drives ``vllm.model_executor.model_loader.flashpack_loader
    .flashpack_weights_iterator`` over a converted checkpoint.

Every subcommand reports peak host RSS and the host page-cache delta
alongside its timings, because on unified-memory parts host bytes are taken
out of the same pool the model and KV cache live in.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
import json
import os
import struct
import subprocess
import sys
import threading
import time

import torch

_PAGE = os.sysconf("SC_PAGE_SIZE")


def _proc_status_kib(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


def _meminfo_kib(field: str) -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


class HostMemSampler:
    """Poll RSS / MemAvailable / Cached in the background.

    ``VmHWM`` alone hides the page cache, and on a unified-memory part the
    page cache is not free either -- it is competing for the same DRAM the
    model sits in.  So sample both.
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._stop = threading.Event()
        self.peak_rss_kib = 0
        self.min_avail_kib = 1 << 62
        self.max_cached_kib = 0
        self._t: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            self.peak_rss_kib = max(self.peak_rss_kib, _proc_status_kib("VmRSS"))
            self.min_avail_kib = min(self.min_avail_kib, _meminfo_kib("MemAvailable"))
            self.max_cached_kib = max(self.max_cached_kib, _meminfo_kib("Cached"))
            self._stop.wait(self.interval)

    @staticmethod
    def _io() -> dict:
        out = {}
        try:
            with open("/proc/self/io") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    out[k] = int(v)
        except OSError:
            pass
        return out

    def __enter__(self):
        self.start_io = self._io()
        self.start_avail_kib = _meminfo_kib("MemAvailable")
        self.start_cached_kib = _meminfo_kib("Cached")
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        assert self._t is not None
        self._t.join()
        self.peak_rss_kib = max(self.peak_rss_kib, _proc_status_kib("VmHWM"))

    def report(self) -> dict:
        end_io = self._io()
        # read_bytes is what actually came off the block device, so it counts
        # a checkpoint that has to be fetched twice.  rchar counts only
        # explicit read() syscalls and therefore misses mmap page faults.
        io = {
            "disk_read_gib": round(
                (end_io.get("read_bytes", 0) - self.start_io.get("read_bytes", 0))
                / 1024**3,
                2,
            ),
            "syscall_read_gib": round(
                (end_io.get("rchar", 0) - self.start_io.get("rchar", 0)) / 1024**3, 2
            ),
        }
        return {
            **io,
            "peak_rss_gib": round(self.peak_rss_kib / 1024**2, 2),
            "vmhwm_gib": round(_proc_status_kib("VmHWM") / 1024**2, 2),
            "memavailable_start_gib": round(self.start_avail_kib / 1024**2, 2),
            "memavailable_min_gib": round(self.min_avail_kib / 1024**2, 2),
            "memavailable_drop_gib": round(
                (self.start_avail_kib - self.min_avail_kib) / 1024**2, 2
            ),
            "page_cache_growth_gib": round(
                (self.max_cached_kib - self.start_cached_kib) / 1024**2, 2
            ),
        }


def drop_caches(paths: list[str] | None = None, global_drop: bool = True) -> None:
    """Evict the checkpoint from the page cache.

    ``posix_fadvise(DONTNEED)`` is applied per file first (it is what a
    non-root process can do), then a global ``drop_caches=3`` via sudo, which
    is the only thing that reliably clears dirty/mapped state.  The
    production init container does the same global drop before vLLM starts.
    """
    if paths:
        for p in paths:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
    if global_drop:
        subprocess.run(["sync"], check=True)
        subprocess.run(
            ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"],
            input=b"3\n",
            stdout=subprocess.DEVNULL,
            check=True,
        )
    time.sleep(1.0)


def select_files(model_dir: str, limit: int | None) -> list[str]:
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise SystemExit(f"no safetensors under {model_dir}")
    if limit:
        files = files[:limit]
    return files


def total_bytes(files: list[str]) -> int:
    return sum(os.path.getsize(f) for f in files)


def shard_stats(files: list[str]) -> dict:
    n_tensors = 0
    n_bytes = 0
    small = 0
    for fn in files:
        with open(fn, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
        hdr.pop("__metadata__", None)
        n_tensors += len(hdr)
        for v in hdr.values():
            sz = v["data_offsets"][1] - v["data_offsets"][0]
            n_bytes += sz
            if sz < 1024 * 1024:
                small += 1
    return {"tensors": n_tensors, "tensor_bytes": n_bytes, "tensors_under_1mib": small}


# --------------------------------------------------------------------------
# raw
# --------------------------------------------------------------------------


def cmd_raw(args) -> None:
    files = select_files(args.model, args.files)
    nbytes = total_bytes(files)
    if not args.no_drop_caches:
        drop_caches(files)

    if args.threads == 1:
        with HostMemSampler() as mem:
            t0 = time.perf_counter()
            buf = bytearray(args.block)
            mv = memoryview(buf)
            for fn in files:
                with open(fn, "rb", buffering=0) as f:
                    while f.readinto(mv):
                        pass
            elapsed = time.perf_counter() - t0
    else:
        import concurrent.futures

        def _read(fn):
            buf = bytearray(args.block)
            mv = memoryview(buf)
            with open(fn, "rb", buffering=0) as f:
                while f.readinto(mv):
                    pass

        with HostMemSampler() as mem:
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(args.threads) as ex:
                list(ex.map(_read, files))
            elapsed = time.perf_counter() - t0

    print(
        json.dumps(
            {
                "mode": "raw",
                "files": len(files),
                "threads": args.threads,
                "block": args.block,
                "gib": round(nbytes / 1024**3, 2),
                "seconds": round(elapsed, 2),
                "gib_per_s": round(nbytes / 1024**3 / elapsed, 2),
                **mem.report(),
            },
            indent=2,
        )
    )


# --------------------------------------------------------------------------
# iterator (the real vLLM generator)
# --------------------------------------------------------------------------


def cmd_iterator(args) -> None:
    import vllm.model_executor.model_loader.weight_utils as wu

    files = select_files(args.model, args.files)
    nbytes = total_bytes(files)
    stats = shard_stats(files)
    if not args.no_drop_caches:
        drop_caches(files)

    dev = torch.device(args.device) if args.device != "none" else None
    if dev is not None and dev.type == "cuda":
        torch.cuda.init()
        torch.cuda.synchronize()

    strategy = None if args.strategy == "default" else args.strategy

    gen_time = 0.0
    consume_time = 0.0
    n = 0
    consumed_bytes = 0
    per_file = []
    dst_buf: torch.Tensor | None = None

    with HostMemSampler() as mem:
        wall0 = time.perf_counter()
        if args.impl == "safetensors":
            it = wu.safetensors_weights_iterator(
                files,
                use_tqdm_on_load=False,
                safetensors_load_strategy=strategy,
                safetensors_prefetch_num_threads=args.prefetch_threads,
            )
        elif args.impl == "multithread":
            it = wu.multi_thread_safetensors_weights_iterator(
                files, use_tqdm_on_load=False, max_workers=args.workers
            )
        elif args.impl == "fastsafetensors":
            it = wu.fastsafetensors_weights_iterator(files, use_tqdm_on_load=False)
        elif args.impl == "instanttensor":
            it = wu.instanttensor_weights_iterator(files, use_tqdm_on_load=False)
        else:
            raise SystemExit(f"unknown impl {args.impl}")
        while True:
            t0 = time.perf_counter()
            try:
                name, param = next(it)
            except StopIteration:
                gen_time += time.perf_counter() - t0
                break
            gen_time += time.perf_counter() - t0

            t1 = time.perf_counter()
            if dev is None:
                # Force the bytes to actually be materialised even if the
                # strategy handed back an mmap-backed view.
                consumed_bytes += param.numel() * param.element_size()
                _ = param[(0,) * param.dim()] if param.dim() else param.item()
            else:
                # A real model's parameters are allocated once by
                # initialize_model and then copied into, so reuse one
                # destination buffer rather than allocating per tensor.
                tbytes = param.numel() * param.element_size()
                if dst_buf is None or dst_buf.numel() < tbytes:
                    dst_buf = None
                    torch.cuda.empty_cache()
                    dst_buf = torch.empty(tbytes, dtype=torch.uint8, device=dev)
                d = dst_buf[:tbytes].view(param.dtype).view(param.shape)
                d.copy_(param, non_blocking=args.non_blocking)
                consumed_bytes += tbytes
            consume_time += time.perf_counter() - t1
            del param
            n += 1
        if dev is not None and dev.type == "cuda":
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            consume_time += time.perf_counter() - t1
        wall = time.perf_counter() - wall0

    print(
        json.dumps(
            {
                "mode": "iterator",
                "impl": args.impl,
                "workers": args.workers,
                "strategy": args.strategy,
                "device": args.device,
                "non_blocking": args.non_blocking,
                "files": len(files),
                "gib_on_disk": round(nbytes / 1024**3, 2),
                "tensors": n,
                "tensors_in_header": stats["tensors"],
                "tensors_under_1mib": stats["tensors_under_1mib"],
                "gib_consumed": round(consumed_bytes / 1024**3, 2),
                "wall_s": round(wall, 2),
                "generator_next_s": round(gen_time, 2),
                "consumer_s": round(consume_time, 2),
                "gib_per_s_wall": round(nbytes / 1024**3 / wall, 2),
                "us_per_tensor_generator": round(gen_time / max(n, 1) * 1e6, 1),
                "us_per_tensor_consumer": round(consume_time / max(n, 1) * 1e6, 1),
                **mem.report(),
            },
            indent=2,
        )
    )
    if per_file:
        print(json.dumps(per_file))


# --------------------------------------------------------------------------
# flashpack
# --------------------------------------------------------------------------


def cmd_flashpack(args) -> None:
    from vllm.model_executor.model_loader.flashpack_loader import (
        flashpack_weights_iterator,
        parse_flashpack_index,
        select_flashpack_parts,
    )

    idx_path = os.path.join(args.model, "model.flashpack.index.json")
    with open(idx_path) as f:
        index = parse_flashpack_index(json.load(f))
    parts = select_flashpack_parts(
        index, lambda _n: True, include_mtp=args.include_mtp
    )
    if args.files:
        parts = parts[: args.files]
    paths = [os.path.join(args.model, p.filename) for p in parts]
    disk_bytes = sum(os.path.getsize(p) for p in paths)
    nbytes = 0
    if not args.no_drop_caches:
        drop_caches(paths)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.init()
        torch.cuda.synchronize()

    gen_time = 0.0
    consume_time = 0.0
    n = 0
    with HostMemSampler() as mem:
        wall0 = time.perf_counter()
        it = flashpack_weights_iterator(
            index,
            parts,
            lambda fn: os.path.join(args.model, fn),
            lambda _n: True,
            device=dev,
            verify_sha256=args.verify_sha256,
        )
        while True:
            t0 = time.perf_counter()
            try:
                name, tensor = next(it)
            except StopIteration:
                gen_time += time.perf_counter() - t0
                break
            gen_time += time.perf_counter() - t0
            # The FlashPack iterator already yields device-resident tensors,
            # so unlike the safetensors path there is no consumer-side H2D to
            # time; the comparison point is bytes-on-GPU either way.
            nbytes += tensor.numel() * tensor.element_size()
            if args.touch:
                # A CPU-target FlashPack read returns lazy mmap views unless
                # FLASHPACK_CPU_PARALLEL_READ=1, so counting bytes would time
                # nothing. Cloning forces materialisation; both variants pay
                # the same one memcpy, so the difference is the fault cost.
                t1 = time.perf_counter()
                tensor.view(torch.uint8).clone()
                consume_time += time.perf_counter() - t1
            del tensor
            n += 1
        if dev.type == "cuda":
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            consume_time += time.perf_counter() - t1
        wall = time.perf_counter() - wall0

    print(
        json.dumps(
            {
                "mode": "flashpack",
                "device": args.device,
                "verify_sha256": args.verify_sha256,
                "parts": len(parts),
                "gib_on_disk": round(disk_bytes / 1024**3, 2),
                "gib_yielded": round(nbytes / 1024**3, 2),
                "tensors": n,
                "wall_s": round(wall, 2),
                "generator_next_s": round(gen_time, 2),
                "consumer_s": round(consume_time, 2),
                "gib_per_s_wall": round(disk_bytes / 1024**3 / wall, 2),
                **mem.report(),
            },
            indent=2,
        )
    )


# --------------------------------------------------------------------------
# engine (real LLM construction, instrumented)
# --------------------------------------------------------------------------

PHASES: dict[str, float] = {}


def _install_instrumentation() -> None:
    """Wrap -- not replace -- the real loader entry points."""
    import vllm.model_executor.model_loader.base_loader as base_loader
    import vllm.model_executor.model_loader.default_loader as default_loader
    import vllm.model_executor.model_loader.utils as ml_utils
    import vllm.model_executor.model_loader.weight_utils as weight_utils

    real_iter = weight_utils.safetensors_weights_iterator

    def timed_iter(*a, **kw):
        gen = real_iter(*a, **kw)
        while True:
            t0 = time.perf_counter()
            try:
                item = next(gen)
            except StopIteration:
                PHASES["weight_iterator_s"] = (
                    PHASES.get("weight_iterator_s", 0.0) + time.perf_counter() - t0
                )
                return
            PHASES["weight_iterator_s"] = (
                PHASES.get("weight_iterator_s", 0.0) + time.perf_counter() - t0
            )
            PHASES["tensors"] = PHASES.get("tensors", 0) + 1
            PHASES["tensor_bytes"] = (
                PHASES.get("tensor_bytes", 0) + item[1].numel() * item[1].element_size()
            )
            t1 = time.perf_counter()
            yield item
            # Time between handing the tensor over and being resumed is the
            # model's weight_loader dispatch plus its host-to-device copy.
            PHASES["weight_loader_s"] = (
                PHASES.get("weight_loader_s", 0.0) + time.perf_counter() - t1
            )

    weight_utils.safetensors_weights_iterator = timed_iter
    default_loader.safetensors_weights_iterator = timed_iter

    real_pwal = ml_utils.process_weights_after_loading

    def timed_pwal(*a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = real_pwal(*a, **kw)
        torch.cuda.synchronize()
        PHASES["process_weights_after_loading_s"] = time.perf_counter() - t0
        return out

    ml_utils.process_weights_after_loading = timed_pwal
    base_loader.process_weights_after_loading = timed_pwal

    real_init = ml_utils.initialize_model

    def timed_init(*a, **kw):
        t0 = time.perf_counter()
        out = real_init(*a, **kw)
        PHASES["initialize_model_s"] = (
            PHASES.get("initialize_model_s", 0.0) + time.perf_counter() - t0
        )
        return out

    ml_utils.initialize_model = timed_init
    base_loader.initialize_model = timed_init

    real_load_model = base_loader.BaseModelLoader.load_model

    def timed_load_model(self, *a, **kw):
        t0 = time.perf_counter()
        out = real_load_model(self, *a, **kw)
        torch.cuda.synchronize()
        PHASES["load_model_total_s"] = time.perf_counter() - t0
        return out

    base_loader.BaseModelLoader.load_model = timed_load_model


def cmd_engine(args) -> None:
    _install_instrumentation()
    from vllm import LLM

    files = select_files(args.model, None)
    if not args.no_drop_caches:
        drop_caches(files)

    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        load_format=args.load_format,
        trust_remote_code=True,
    )
    if args.strategy != "default":
        kwargs["safetensors_load_strategy"] = args.strategy
    if args.extra_config:
        kwargs["model_loader_extra_config"] = json.loads(args.extra_config)

    llm = None
    ctor_error = None
    with HostMemSampler() as mem:
        t0 = time.perf_counter()
        try:
            llm = LLM(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # Everything after load_model (the profile run, cudagraph capture)
            # can fail on a box whose compiled kernels do not match the model,
            # and none of it is part of what is being measured. The load
            # phases are already recorded by then.
            ctor_error = f"{type(exc).__name__}: {exc}"
        total = time.perf_counter() - t0

    out = {
        "mode": "engine",
        "model": args.model,
        "tp": args.tp,
        "load_format": args.load_format,
        "strategy": args.strategy,
        "llm_ctor_total_s": round(total, 2),
        "ctor_error": ctor_error,
        "gib_on_disk": round(total_bytes(files) / 1024**3, 2),
        **{
            k: (round(v, 3) if isinstance(v, float) else v) for k, v in PHASES.items()
        },
        **mem.report(),
    }
    out["gib_yielded"] = round(PHASES.get("tensor_bytes", 0) / 1024**3, 2)
    print("BENCH_RESULT " + json.dumps(out, indent=2))
    del llm


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", required=True)
        sp.add_argument("--files", type=int, default=None)
        sp.add_argument("--no-drop-caches", action="store_true")

    sp = sub.add_parser("raw")
    common(sp)
    sp.add_argument("--threads", type=int, default=1)
    sp.add_argument("--block", type=int, default=8 << 20)
    sp.set_defaults(func=cmd_raw)

    sp = sub.add_parser("iterator")
    common(sp)
    sp.add_argument(
        "--strategy",
        default="lazy",
        choices=["default", "lazy", "eager", "pinned", "prefetch"],
    )
    sp.add_argument("--device", default="none")
    sp.add_argument("--non-blocking", action="store_true")
    sp.add_argument("--prefetch-threads", type=int, default=8)
    sp.add_argument(
        "--impl",
        default="safetensors",
        choices=["safetensors", "multithread", "fastsafetensors", "instanttensor"],
        help="which real vLLM weight iterator to drive",
    )
    sp.add_argument("--workers", type=int, default=8)
    sp.set_defaults(func=cmd_iterator)

    sp = sub.add_parser("flashpack")
    common(sp)
    sp.add_argument("--device", default="cuda:0")
    sp.add_argument("--verify-sha256", action="store_true")
    sp.add_argument("--include-mtp", action="store_true")
    sp.add_argument("--touch", action="store_true")
    sp.set_defaults(func=cmd_flashpack)

    sp = sub.add_parser("engine")
    sp.add_argument("--model", required=True)
    sp.add_argument("--no-drop-caches", action="store_true")
    sp.add_argument("--tp", type=int, default=1)
    sp.add_argument("--max-model-len", type=int, default=2048)
    sp.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    sp.add_argument("--load-format", default="auto")
    sp.add_argument(
        "--strategy",
        default="default",
        choices=["default", "lazy", "eager", "pinned", "prefetch"],
    )
    sp.add_argument("--extra-config", default=None)
    sp.set_defaults(func=cmd_engine)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
