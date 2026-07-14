# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag

    # DSpark uses non-causal attention.
    causal = False
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=not causal,
            backend=speculative_config.attention_backend,
        ),
    )

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    # Skip embedding sharing under PP -- each rank owns its own embedding
    # (mirrors eagle/utils.py and dflash/utils.py). Under PP>1 the target's
    # REAL embed_tokens lives on get_pp_group().is_first_rank, but this
    # draft's layers/heads are colocated on is_last_rank (see
    # DSparkDeepseekV4Model's _owns_dspark_layers in nvidia/dspark.py) --
    # those are different ranks whenever pipeline_parallel_size > 1, so
    # aliasing would silently wire in either a PPMissingLayer or a
    # first-rank-only tensor. The draft already constructs its own
    # VocabParallelEmbedding unconditionally (loaded from the checkpoint like
    # any other parameter), so simply not aliasing is correct, not degraded.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None)
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            draft_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
