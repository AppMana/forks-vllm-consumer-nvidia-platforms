# DeepSeek V4 Flash Vision on Ampere

Convert the vision checkpoint itself, including its native three-stage DSpark
weights. All 43 decoder layers and three draft stages contain image router biases.

```bash
.venv/bin/python tools/ampere/dsv4_requant_checkpoint.py \
  --src /path/to/DeepSeek-V4-Flash-Vision-Exp \
  --dst /path/to/vision-int4-int8 --device cuda:0 \
  --expert-format int4 --expert-int4-scale-mode mse \
  --dense-int8-strategy channel --vision-format int8-imma
```

`--vision-format auto` selects IMMA for a vision checkpoint with INT4 experts.
Use `--vision-format bf16` for an accuracy reference with the same language
quantization. The original BF16 tower remains compatible with the source's
global FP8 configuration.

The patch projection, attention projections, MLPs, and aligner use signed INT8
weights and dynamic INT8 activations, with separate scales for each group of 32
input channels. Scales are FP32. Projection kernels accumulate integer tensor
core products in INT32 and rescale into FP32. Vision attention uses integer
tensor cores for both QK and PV, with FP32 online softmax and accumulation.
Normalization, RoPE, residuals, learned image vectors, and nonlinear operations
retain floating-point computation. Both kernels require SM80 or newer.

The output configuration explicitly selects the vision kernels. The index
includes every new scale and materialized native draft tensor. Shards retain
their PP ownership and contain at most 4 GiB of tensor data; safetensors headers
add a small amount to the file size.

Only PP rank 0 owns the vision tower, aligner, and learned image vectors. Raw
image sentinel token IDs must remain available on all decoder ranks for image
router biases. The final rank owns the output head and the DSpark stages.

For 43 layers on PP11, an initial partition is
`4,4,4,4,4,4,4,4,5,5,1`, set through `VLLM_PP_LAYER_PARTITION`.
Measure peak memory with the intended image count, concurrency, KV allocation,
and graph configuration before accepting it. Checkpoint bytes alone do not
capture runtime packing, shared draft parameters, or encoder workspaces.
Run `.venv/bin/python -m vllm.layer_partition partition --config /path/to/config.json
--pp-size 11 --memory-profile profile.json` to solve a contiguous partition from `layer_bytes`, `rank_overhead_bytes`, and
`max_rank_bytes`; include runtime reservations in the overhead or budget.

Use `--limit-mm-per-prompt '{"image":8}' --max-num-seqs 8` for eight images and
eight concurrent requests. Select seven probabilistic draft tokens with
`--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'`.
The checkpoint's trained block size remains unchanged; the serving option
controls speculative depth.

Before deployment, compare full-model answers against the BF16-vision variant,
verify PP image/text and draft decoding, and test cache isolation for different
images with identical text. Full-image identities and placeholder positions
must participate in the external cache key without changing router token IDs.
