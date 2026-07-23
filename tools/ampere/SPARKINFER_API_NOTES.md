# sparkinfer API notes for the sm12x (GB10) integration

Captured live from spark-5867 (`~/workspace/venv`), sparkinfer 1.0.1 installed from
`git+https://github.com/local-inference-lab/sparkinfer` @ cc9b476e (PyPI `sparkinfer`
0.0.1 is an explicit name-squat placeholder — never install from PyPI). Deps:
`nvidia-cutlass-dsl==4.6.0`, torch 2.13.0+cu130. `is_supported()` returns True on
GB10 (sm_121, driver 580.142); a CuTe-DSL kernel JIT-compiled and ran in ~2s.
`comm.pcie` JITs via torch cpp_extension and requires nvcc at runtime — the GB10
container must ship the CUDA toolchain.

## Op inventory (list_ops, 17 ops, all archs sm120a/sm121a)

- attention: `paged`, `sparse_mla` (recipes dsv4, glm_nsa), `compressed_mla` (dsv4),
  `nsa_indexer` (dsv4, glm_nsa, msa), `varlen`
- gemm: `blockscaled` (nvfp4/mxfp4/mxfp8), `block_fp8_linear`, `bmm`, `mxfp8_linear`,
  `mla_query_projection` (fused MXFP8 query BMM + RoPE append + optional static E4M3
  quant), `wo_projection` (incl. inv-rope path)
- moe: `fused_moe` (recipes nvfp4, mxfp4, w4a8_mx, w4a8_nvfp4, w4a16 — bf16
  activations), `ep_moe` (w4a16)
- norm: `mhc`; quantization: `nvfp4`, `mxfp8`; comm: `pcie` (Oneshot/Dma AllReduce,
  TwoShotReduceScatter, DcpAllToAll; requires multi_gpu, nvcc at runtime)

All planned-style ops follow plan(Caps) -> bind(Plan, **tensors) -> run_*; bind is
views-only (CUDA-graph-capture safe).

## sparse_mla (the Phase 2c decode/prefill target)

```
Caps(*, device, num_q_heads, max_q_rows, max_width,
     dtype=torch.bfloat16, kv_dtype=torch.bfloat16,
     head_dim=576, v_head_dim=512,                      # DSV4 shapes are the defaults
     mode: 'decode'|'extend'|'verify'|'draft_extend' = 'decode',
     max_batch=None, max_kv_rows=0, max_page_table_width=None,
     max_chunks_per_row=64, max_q_chunks=None, page_size=64,
     head_major_output=False)

Binding(*, scratch, q, selected_indices, cache_seqlens_int32, nsa_cache_seqlens_int32)

DecodeMetadata(page_table_1, cache_seqlens_int32, nsa_cache_seqlens_int32, max_seq_len_k)
ExtendMetadata(selected_token_offsets, cache_seqlens_int32, nsa_cache_seqlens_int32,
               nsa_cu_seqlens_q, nsa_cu_seqlens_k, max_seq_len_q, max_seq_len_k,
               mode: 'extend'|'verify'|'target_verify'|'draft_extend' = 'extend')

run_decode(*, q_all=None, kv_cache, page_table_1=None, cache_seqlens_int32=None,
           nsa_cache_seqlens_int32=None, binding=None, sm_scale, latent_scale=1.0,
           v_head_dim=None, return_lse=False, lse_scale='base2'|'natural',
           attn_sink=None,                               # our layer's attn_sink maps here
           identity_page_table=False, backend=None, forced_num_splits=None,
           scale_format=None, fp8_rope=None) -> out | (out, lse)

run_extend(*, q_all=None, kv_cache, selected_token_offsets=None,
           cache_seqlens_int32=None, nsa_cache_seqlens_int32=None, binding=None,
           sm_scale, latent_scale=1.0, v_head_dim=None, return_lse=False,
           lse_scale=..., identity_page_table=False, scale_format=None,
           fp8_rope=None) -> out | (out, lse)
```

Notes for the vLLM layer:

- `mode='verify'` / `'draft_extend'` exist as first-class modes — directly relevant
  to composing with DSpark speculative decoding (target-verify batches).
- `run_extend` has NO `attn_sink` parameter (decode does). Prefill attn-sink
  handling needs a check against the kernel's semantics before relying on it.
- `kv_dtype` supports fp8_e4m3 (matches `fp8_ds_mla` cache); `scale_format` /
  `fp8_rope` knobs control the fp8 cache interpretation — verify against our
  656-byte fp8_ds_mla token row layout when wiring `_forward_decode`.
- `Scratch` is component-owned VIEWS over caller-owned storage (docstring:
  "NEVER a SPARKINFERAttentionWorkspace") — integrates with our
  `current_workspace_manager()` pattern.

## nsa_indexer highlights

dsv4 recipe; paged + contiguous bindings; `quantize_q_fp8`, `logits_paged`,
`topk_blocks`, `topk_tiled`, `q2k_indices_{decode,prefill}`, and a persistent
top-k 2048 path (`plan_persistent_topk2048` / `run_persistent_topk2048`,
`supports_persistent_topk2048`). `INDEX_HEAD_DIM`, `PAGED_INDEX_PAGE_SIZE`
constants. Candidate replacement/complement for our streaming prefill top-k;
compare in Phase 6 microbenches.

## fused_moe highlights

`plan_weights` / `prepare_weights` + `ExecutionPlan`, `run` / `run_sparse`,
`route` / `route_topk`. Recipes include `nvfp4` and `w4a8_nvfp4` (bf16
activations) — consumes the modelopt-NVFP4 checkpoint format for Phase 2d via
the `NvFp4MoeBackend.FLASHINFER_B12X` seam (`flashinfer_b12x_moe.py`).
