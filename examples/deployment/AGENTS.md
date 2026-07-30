# DSV4 INT4/INT8 deployment acceptance

Verify in this order:

1. Every rank loads the selected checkpoint revision and image digest.
2. Startup logs show the expected kernel mapping and AllSpark selection.
3. `/health` returns HTTP 200 with no rank restart.
4. Chat generation, reasoning-time tool calls, and code execution return
   correct results.
5. Node logs remain free of Xid, illegal-address, OOM, and JIT compilation
   during measured requests.
6. Run C=1 and C=2 at short and long input lengths, then run the context ladder.

Do not publish throughput from a run that fails output correctness. Prefer the
DSV4 needle benchmark, which measures throughput and verifies a request-unique
needle.

## Serving constraints

- DSV4 and DSpark must use model runner v2.
- Benchmarks use async scheduling and CUDA graphs. Use eager mode only for
  diagnosis.
- On unified-memory GB10, explicit KV-cache bytes add to the GPU-utilization
  allocation. Leave enough host memory to prevent swapping.
- Keep weights in their per-expert, separate-projection checkpoint layout.
  Fuse them while loading.
- `--max-num-seqs` changes activation and workspace reservations as well as
  scheduling.
- Compiler caches must be persistent and rank-local. Do not place TileLang,
  Triton, TorchInductor, DeepGEMM, FlashInfer, or `TMPDIR` under `/tmp` in
  Kubernetes.
