# Investigation: do chunked-prefill microbatches pipeline through PP stages?
#
# Runs one long prompt through chunked prefill and records, on the driver:
#   - "sched" events: wall time each scheduler_output is produced (chunk submitted)
#   - "done"  events: wall time each chunk's model output returns (update_from_output)
#
# If chunks pipeline: all N "sched" events cluster at the start (batch queue
# fills to pp_size) and "done" events tick at ~max(stage) intervals.
# If serial: "done" events tick at ~sum(stage) intervals (PP=2: ~= PP=1 chunk time).
#
# Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 (set below) so EngineCore runs
# in-process and class monkeypatches apply.

import argparse
import os
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

EVENTS: list[tuple[str, float, int]] = []


def install_probes():
    from vllm.v1.core.sched.scheduler import Scheduler

    orig_schedule = Scheduler.schedule
    orig_update = Scheduler.update_from_output

    def schedule(self, *a, **k):
        out = orig_schedule(self, *a, **k)
        if out.total_num_scheduled_tokens > 0:
            EVENTS.append(("sched", time.perf_counter(), out.total_num_scheduled_tokens))
        return out

    def update_from_output(self, scheduler_output, model_output):
        if scheduler_output.total_num_scheduled_tokens > 0:
            EVENTS.append(
                ("done", time.perf_counter(), scheduler_output.total_num_scheduled_tokens)
            )
        return orig_update(self, scheduler_output, model_output)

    Scheduler.schedule = schedule
    Scheduler.update_from_output = update_from_output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    p.add_argument("--pp", type=int, default=2)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--prompt-tokens", type=int, default=12288)
    p.add_argument("--async-sched", action="store_true")
    p.add_argument("--gpu-util", type=float, default=0.6)
    args = p.parse_args()

    install_probes()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        pipeline_parallel_size=args.pp,
        distributed_executor_backend="mp" if args.pp > 1 else None,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.chunk,
        max_num_seqs=8,
        max_model_len=min(args.prompt_tokens + 512, 8192)
        if "SmolLM2" in args.model
        else args.prompt_tokens + 512,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_util,
        async_scheduling=args.async_sched,
        disable_log_stats=True,
    )

    sp = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)

    # Warmup (lazy init paths), then clear the event log.
    llm.generate(TokensPrompt(prompt_token_ids=[100] * (2 * args.chunk)), sp)
    EVENTS.clear()

    t0 = time.perf_counter()
    llm.generate(TokensPrompt(prompt_token_ids=[100] * args.prompt_tokens), sp)
    t1 = time.perf_counter()

    print("\n================ EVENT LOG ================")
    print(f"config: pp={args.pp} chunk={args.chunk} prompt={args.prompt_tokens} "
          f"async_sched={args.async_sched} model={args.model}")
    print(f"total generate() wall time: {t1 - t0:.3f}s")
    prev_done = None
    prev_sched = None
    for kind, t, ntok in EVENTS:
        rel = t - t0
        if kind == "sched":
            delta = "" if prev_sched is None else f" (+{(t - prev_sched) * 1e3:7.1f}ms)"
            prev_sched = t
        else:
            delta = "" if prev_done is None else f" (+{(t - prev_done) * 1e3:7.1f}ms)"
            prev_done = t
        print(f"  t={rel * 1e3:9.1f}ms  {kind:5s}  ntok={ntok:5d}{delta}")

    # Summary: inter-done intervals during prefill (ntok == chunk).
    dones = [t for kind, t, ntok in EVENTS if kind == "done" and ntok == args.chunk]
    if len(dones) > 2:
        deltas = [(b - a) * 1e3 for a, b in zip(dones, dones[1:])]
        deltas_ss = sorted(deltas)
        n = len(deltas)
        print(f"\nprefill inter-chunk 'done' intervals (n={n}): "
              f"min={deltas_ss[0]:.1f}ms p50={deltas_ss[n // 2]:.1f}ms "
              f"max={deltas_ss[-1]:.1f}ms mean={sum(deltas) / n:.1f}ms")
    # Max in-flight depth: scheds issued minus dones returned, over time.
    depth = 0
    max_depth = 0
    for kind, _, ntok in EVENTS:
        depth += 1 if kind == "sched" else -1
        max_depth = max(max_depth, depth)
    print(f"max in-flight scheduled-but-not-updated batches: {max_depth}")


if __name__ == "__main__":
    main()
