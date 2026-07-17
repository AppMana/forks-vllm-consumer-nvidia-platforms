# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-name remapping for the DSpark draft loader.

Grafted Base+DSpark checkpoints (e.g. appmana/deepseek-v4-int4-int8) ship
per-stage ``mtp.N.emb.tok_emb.weight`` / ``mtp.N.head.weight`` copies that are
byte-identical to the target's embed/head. The remap must load stage 0's
embedding into the draft's own model-level VocabParallelEmbedding (required
under PP>1, where target-aliasing is skipped) and drop everything redundant.
"""

import pytest

from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4ForCausalLM


def remap(name: str) -> str | None:
    # _remap_dspark_name reads nothing off self; call it unbound so the test
    # needs no VllmConfig/GPU to instantiate the draft.
    return DSparkDeepseekV4ForCausalLM._remap_dspark_name(None, name)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Stage 0's embedding copy fills the draft's own embedding.
        ("mtp.0.emb.tok_emb.weight", "model.embed_tokens.weight"),
        # Redundant per-stage embedding copies are dropped.
        ("mtp.1.emb.tok_emb.weight", None),
        ("mtp.2.emb.tok_emb.weight", None),
        # lm_head is aliased from the target (same last rank under any PP).
        ("mtp.0.head.weight", None),
        ("mtp.2.head.weight", None),
        # Confidence head is not wired into inference.
        ("mtp.0.confidence_head.proj.weight", None),
        # Non-mtp weights belong to the target model.
        ("embed.weight", None),
        ("head.weight", None),
        ("layers.0.attn.wq_a.weight", None),
        # Head-stack params live at model level.
        ("mtp.2.norm.weight", "model.norm.weight"),
        ("mtp.2.hc_head_fn", "model.hc_head_fn"),
        ("mtp.2.markov_head.markov_w1.weight", "model.markov_head.markov_w1.weight"),
        ("mtp.0.main_proj.weight", "model.main_proj.weight"),
        ("mtp.0.main_norm.weight", "model.main_norm.weight"),
        # Everything else is a per-stage decoder block.
        ("mtp.1.attn.wq_a.weight", "model.layers.1.attn.wq_a.weight"),
        ("mtp.2.ffn.experts.7.w1.scale", "model.layers.2.ffn.experts.7.w1.scale"),
        ("mtp.0.hc_attn_fn", "model.layers.0.hc_attn_fn"),
        ("mtp.1.attn_norm.weight", "model.layers.1.attn_norm.weight"),
    ],
)
def test_remap_dspark_name(name: str, expected: str | None) -> None:
    assert remap(name) == expected
