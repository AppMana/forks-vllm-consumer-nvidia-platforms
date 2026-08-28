# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint selection and name remapping for the DSpark draft loader.

Grafted Base+DSpark checkpoints (e.g. appmana/deepseek-v4-int4-int8) ship
per-stage ``mtp.N.emb.tok_emb.weight`` / ``mtp.N.head.weight`` copies that are
byte-identical to the target's embed/head. The remap must load stage 0's
embedding into the draft's own model-level VocabParallelEmbedding (required
under PP>1, where target-aliasing is skipped) and drop everything redundant.
"""

import json
from types import SimpleNamespace

import pytest
from torch import nn

from vllm.models.deepseek_v4.nvidia import dspark
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


@pytest.mark.parametrize(
    ("weight_map", "expected"),
    [
        (
            {
                "model.layers.0.weight": "model-00001-of-00048.safetensors",
                "mtp.0.attn.wq_a.weight": "model-00046-of-00048.safetensors",
                "mtp.1.attn.wq_a.weight": "model-00047-of-00048.safetensors",
                "mtp.2.attn.wq_a.weight": "model-00048-of-00048.safetensors",
                "mtp.0.emb.tok_emb.weight": "model-mtp-shared.safetensors",
                "mtp.1.head.weight": "model-mtp-shared.safetensors",
            },
            [
                "model-00046-of-00048.safetensors",
                "model-00047-of-00048.safetensors",
                "model-00048-of-00048.safetensors",
                "model-mtp-shared.safetensors",
            ],
        ),
        (
            {
                "mtp.0.attn.wq_a.weight": "model-mtp-dspark.safetensors",
                "mtp.2.norm.weight": "model-mtp-dspark.safetensors",
                "mtp.0.emb.tok_emb.weight": "model-mtp-shared.safetensors",
            },
            ["model-mtp-dspark.safetensors", "model-mtp-shared.safetensors"],
        ),
    ],
)
def test_dspark_selects_every_indexed_loadable_shard(
    tmp_path, monkeypatch, weight_map: dict[str, str], expected: list[str]
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    for shard in set(weight_map.values()):
        (tmp_path / shard).touch()

    monkeypatch.setattr(dspark, "DSparkDeepseekV4Model", lambda **_: nn.Identity())
    monkeypatch.setattr(
        dspark, "ParallelLMHead", lambda *_args, **_kwargs: nn.Identity()
    )
    monkeypatch.setattr(dspark, "LogitsProcessor", lambda *_args, **_kwargs: object())

    draft_model_config = SimpleNamespace(
        model=str(tmp_path), hf_config=SimpleNamespace(vocab_size=1, hidden_size=1)
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=draft_model_config,
        ),
        quant_config=SimpleNamespace(weight_block_size=None),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
    )

    model = DSparkDeepseekV4ForCausalLM(vllm_config=vllm_config)

    assert model.allow_patterns_overrides == expected
