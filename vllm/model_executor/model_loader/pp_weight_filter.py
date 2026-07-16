# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filter out non-local pipeline-parallel layer weights during loading to
avoid redundant I/O.

In a PP deployment each rank only holds a contiguous slice of the model's
hidden layers (see vllm.distributed.utils.get_pp_indices /
vllm.model_executor.models.utils.make_layers). Without this filter,
safetensors_weights_iterator calls f.get_tensor() for every per-layer weight
in every checkpoint shard regardless of which PP rank is loading -- unlike
ep_weight_filter.should_skip_weight, which already skips non-local *expert*
tensors before reading them from disk, there was no equivalent skip for
non-local *layer* tensors, so every PP rank downloaded close to the full
checkpoint instead of its own 1/pp_size share.
"""

import regex as re

# Matches the standard vLLM per-layer weight naming convention produced by
# vllm.model_executor.models.utils.make_layers(prefix=f"{prefix}.layers"):
# e.g. "model.layers.18.self_attn.q_proj.weight" (nested under a model
# prefix) or "layers.18.attn.attn_sink" (DeepSeek-V4's int4/int8 checkpoint,
# which has NO leading "model." prefix -- "layers.N." starts the string
# directly). Anchored on "layers." preceded by either a literal dot or the
# start of the string -- NOT a bare "layers." substring match, which would
# also false-match an unrelated identifier like "sublayers.0.x". This mirrors
# ep_weight_filter's ".experts." anchor, generalized to also cover the
# start-of-string case.
_LAYER_ID_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def parse_layer_id(weight_name: str) -> int | None:
    """Return the hidden-layer index embedded in *weight_name*, or ``None``
    if the weight is not associated with a specific numbered layer (e.g.
    embedding, final norm, lm_head -- these are always dense/shared and must
    never be skipped)."""
    m = _LAYER_ID_RE.search(weight_name)
    return int(m.group(1)) if m else None


def should_skip_pp_weight(
    weight_name: str,
    local_layer_range: tuple[int, int] | None,
) -> bool:
    """Return ``True`` if *weight_name* is a per-layer weight whose layer
    index falls outside this rank's local ``[start, end)`` layer range and
    should be skipped during loading.

    ``local_layer_range`` is ``None`` when pipeline parallelism is not
    active (pp_size <= 1), in which case nothing is ever skipped.
    """
    if local_layer_range is None:
        return False
    lid = parse_layer_id(weight_name)
    if lid is None:
        # Not a per-layer weight (embedding / final norm / lm_head / etc.)
        # -> always local, never skip.
        return False
    start, end = local_layer_range
    return not (start <= lid < end)
