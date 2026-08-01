"""GB10 microbenchmark for the DSV4 decode-width regression.

Run this file directly on an sm_121 host to print CUDA-event timings for the
exact DSpark indexer row count at a 64K request with a 250K admission limit.
"""

import torch

ROWS = 6  # one target token plus five DSpark draft tokens
HEADS = 64
HEAD_DIM = 128
PAGE_SIZE = 64
LIVE_WIDTH = 16_384  # 64K context compressed by four
CONFIGURED_WIDTH = 62_500  # 250K admission limit compressed by four


def _make_inputs(width: int):
    pages = (width + PAGE_SIZE - 1) // PAGE_SIZE
    token_bytes = HEAD_DIM + torch.float32.itemsize
    cache = torch.zeros(
        pages,
        PAGE_SIZE,
        1,
        token_bytes,
        device="cuda",
        dtype=torch.uint8,
    )
    flat = cache.view(pages, -1)
    value_bytes = PAGE_SIZE * HEAD_DIM
    flat[:, :value_bytes] = torch.randint(
        0, 255, (pages, value_bytes), device="cuda", dtype=torch.uint8
    )
    flat[:, value_bytes:].view(torch.float32).fill_(0.01)
    q = torch.randint(
        -127,
        128,
        (ROWS, HEADS, HEAD_DIM),
        device="cuda",
        dtype=torch.int8,
    )
    weights = torch.rand(ROWS, HEADS, device="cuda", dtype=torch.float32)
    page_table = (
        torch.arange(pages, device="cuda", dtype=torch.int32)
        .expand(ROWS, -1)
        .contiguous()
    )
    lens = torch.full((ROWS,), LIVE_WIDTH, device="cuda", dtype=torch.int32)
    return q, weights, cache, page_table, lens


def _time_paged_logits(width: int, repeats: int = 5) -> float:
    from sparkinfer.attention.nsa_indexer.kernel import run_paged_logits_kernel

    q, weights, cache, page_table, lens = _make_inputs(width)

    def run():
        return run_paged_logits_kernel(
            q_fp8=q,
            weights=weights,
            index_k_cache=cache,
            real_page_table=page_table,
            seqlens_per_query=lens,
            page_size=PAGE_SIZE,
        )

    run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        out = run()
    end.record()
    end.synchronize()
    # SparkInfer pads its physical output width to a kernel tile; vLLM slices
    # this view back to the requested logical width.
    assert out.shape[0] == ROWS and out.shape[1] >= width
    return start.elapsed_time(end) / repeats


def _time_triton_paged_logits(width: int, repeats: int = 5) -> float:
    from vllm.models.deepseek_v4.nvidia_imma import triton_kernels

    q, weights, cache, page_table, lens = _make_inputs(width)
    triton_kernels.indexer_cache_is_int8 = lambda: True

    def run():
        return triton_kernels.fp8_paged_mqa_logits_triton(
            q.view(1, ROWS, HEADS, HEAD_DIM),
            cache,
            weights,
            lens.view(1, ROWS),
            page_table[:1],
            max_model_len=width,
            token_count=width,
        )

    run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        out = run()
    end.record()
    end.synchronize()
    assert out.shape == (ROWS, width)
    return start.elapsed_time(end) / repeats


def _time_decode_topk(width: int, repeats: int = 20) -> float:
    import vllm._custom_ops  # noqa: F401 - registers torch.ops._C

    logits = torch.randn(ROWS, width, device="cuda", dtype=torch.float32)
    lens = torch.full((ROWS, 1), LIVE_WIDTH, device="cuda", dtype=torch.int32)
    indices = torch.empty(ROWS, 512, device="cuda", dtype=torch.int32)

    def run():
        torch.ops._C.top_k_per_row_decode(
            logits,
            1,
            lens,
            indices,
            ROWS,
            logits.stride(0),
            logits.stride(1),
            512,
        )

    run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


if __name__ == "__main__":
    if torch.cuda.get_device_capability() != (12, 1):
        raise SystemExit("sm_121 required")
    # First pass initializes the CUDA context and both kernel variants. Report
    # subsequent alternating passes so first-use migration/JIT is not confused
    # with steady-state kernel work.
    for width in (LIVE_WIDTH, CONFIGURED_WIDTH):
        _time_paged_logits(width, repeats=1)
        _time_triton_paged_logits(width, repeats=1)
        _time_decode_topk(width, repeats=1)
    for pass_id in range(3):
        for width in (
            (LIVE_WIDTH, CONFIGURED_WIDTH)
            if pass_id % 2 == 0
            else (CONFIGURED_WIDTH, LIVE_WIDTH)
        ):
            print(
                f"pass={pass_id} width={width} rows={ROWS} "
                f"paged_logits_ms={_time_paged_logits(width):.3f} "
                f"triton_paged_logits_ms={_time_triton_paged_logits(width):.3f} "
                f"topk_ms={_time_decode_topk(width):.3f}"
            )
