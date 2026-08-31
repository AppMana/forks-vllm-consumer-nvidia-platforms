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
import torch
from torch import nn

from vllm.model_executor.models import qwen3_dspark
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
        # The 0731 confidence head is a model-level draft head.
        (
            "mtp.0.confidence_head.proj.weight",
            "model.confidence_head.proj.weight",
        ),
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
    ("weight_map", "expected_files", "expected_names"),
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
            {
                "mtp.0.attn.wq_a.weight",
                "mtp.1.attn.wq_a.weight",
                "mtp.2.attn.wq_a.weight",
            },
        ),
        (
            {
                "mtp.0.attn.wq_a.weight": "model-mtp-dspark.safetensors",
                "mtp.2.norm.weight": "model-mtp-dspark.safetensors",
                "mtp.0.emb.tok_emb.weight": "model-mtp-shared.safetensors",
            },
            ["model-mtp-dspark.safetensors", "model-mtp-shared.safetensors"],
            {"mtp.0.attn.wq_a.weight", "mtp.2.norm.weight"},
        ),
        (
            {
                "mtp.0.ffn.experts.7.w1.weight": "model-mtp-experts.safetensors",
                "mtp.0.ffn.shared_experts.w1.weight": "model-mtp-shared.safetensors",
                "mtp.0.confidence_head.proj.weight": "model-mtp-confidence.safetensors",
            },
            [
                "model-mtp-confidence.safetensors",
                "model-mtp-experts.safetensors",
                "model-mtp-shared.safetensors",
            ],
            {
                "mtp.0.ffn.experts.7.w1.weight",
                "mtp.0.ffn.shared_experts.w1.weight",
                "mtp.0.confidence_head.proj.weight",
            },
        ),
    ],
)
def test_dspark_selects_every_indexed_loadable_shard(
    tmp_path,
    monkeypatch,
    weight_map: dict[str, str],
    expected_files: list[str],
    expected_names: set[str],
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

    assert model.allow_patterns_overrides == expected_files
    assert model._expected_dspark_checkpoint_names == expected_names


def test_dspark_manifest_defers_routed_experts_when_ep_filter_can_hide_them(
    tmp_path,
) -> None:
    weight_map = {
        "mtp.0.ffn.experts.7.w1.weight": "model-mtp.safetensors",
        "mtp.0.ffn.shared_experts.w1.weight": "model-mtp.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (tmp_path / "model-mtp.safetensors").touch()

    manifest = dspark._indexed_dspark_weight_manifest(
        str(tmp_path), require_routed_experts=False
    )

    assert manifest is not None
    assert manifest[1] == {"mtp.0.ffn.shared_experts.w1.weight"}


def _make_weight_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    owns_layers: bool = True,
    include_embed: bool = False,
    include_confidence: bool = False,
    shares_target_embed: bool = False,
    expected_checkpoint_names: set[str] | None = None,
) -> DSparkDeepseekV4ForCausalLM:
    monkeypatch.setattr(
        dspark,
        "fused_moe_make_expert_params_mapping",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 0)

    model = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model._owns_dspark_layers = owns_layers
    model.model.layers = [
        SimpleNamespace(
            ffn=SimpleNamespace(
                use_mega_moe=False,
                finalize_mega_moe_weights=lambda: None,
            )
        )
    ]
    model.model.main_norm = nn.Module()
    model.model.main_norm.weight = nn.Parameter(torch.empty(1))
    if include_embed:
        model.model.embed_tokens = nn.Embedding(1, 1)
    if include_confidence:
        model.model.confidence_head = nn.Module()
        model.model.confidence_head.proj = nn.Linear(1, 1, bias=False)
        model.model.confidence_head.proj.weight.is_checkpoint_optional = True
    model.lm_head = nn.Linear(1, 1, bias=False)
    model.config = SimpleNamespace(
        n_routed_experts=0,
        num_attention_heads=1,
        expert_dtype="int4",
    )
    model.quant_config = SimpleNamespace(weight_block_size=None)
    model.pad_shared_expert = False
    model._shares_target_embed_tokens = shares_target_embed
    model._expected_dspark_checkpoint_names = expected_checkpoint_names
    return model


def test_dspark_rejects_missing_owned_checkpoint_parameter(monkeypatch) -> None:
    model = _make_weight_loading_model(monkeypatch)

    with pytest.raises(ValueError, match="model.main_norm.weight"):
        model.load_weights([])


def test_dspark_allows_runtime_parameter_without_checkpoint_source(monkeypatch) -> None:
    model = _make_weight_loading_model(monkeypatch)
    runtime_scale = nn.Parameter(torch.full((1,), float("nan")))
    runtime_scale.is_checkpoint_optional = True
    model.model.main_norm.register_parameter("weight_scale_inv", runtime_scale)

    loaded = model.load_weights([("mtp.0.main_norm.weight", torch.ones(1))])

    assert loaded == {"model.main_norm.weight"}


def test_dspark_loads_confidence_head_while_allowing_target_aliases(
    monkeypatch,
) -> None:
    model = _make_weight_loading_model(
        monkeypatch,
        include_embed=True,
        include_confidence=True,
        shares_target_embed=True,
    )

    loaded = model.load_weights(
        [
            ("mtp.0.main_norm.weight", torch.ones(1)),
            ("mtp.0.confidence_head.proj.weight", torch.ones(1)),
        ]
    )

    assert loaded == {
        "model.main_norm.weight",
        "model.confidence_head.proj.weight",
    }


def test_dspark_disables_confidence_for_older_checkpoint(monkeypatch) -> None:
    model = _make_weight_loading_model(monkeypatch, include_confidence=True)

    model.load_weights([("mtp.0.main_norm.weight", torch.ones(1))])

    assert model.model.confidence_head is None


def test_dspark_confidence_uses_hidden_and_markov_state(monkeypatch) -> None:
    class SumProjection(nn.Module):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.sum(dim=-1, keepdim=True)

    monkeypatch.setattr(qwen3_dspark, "ReplicatedLinear", SumProjection)
    head = qwen3_dspark.DSparkConfidenceHead(3, prefix="confidence")
    hidden = torch.tensor([[1.0, 2.0]])
    markov = torch.tensor([[3.0]])

    assert head(hidden, markov).tolist() == [6.0]


def test_dspark_requires_pp_local_embedding_when_it_cannot_be_aliased(
    monkeypatch,
) -> None:
    model = _make_weight_loading_model(
        monkeypatch, include_embed=True, shares_target_embed=False
    )

    with pytest.raises(ValueError, match="model.embed_tokens.weight"):
        model.load_weights([("mtp.0.main_norm.weight", torch.ones(1))])


def test_dspark_rejects_missing_source_component_of_loaded_parameter(
    monkeypatch,
) -> None:
    model = _make_weight_loading_model(
        monkeypatch,
        expected_checkpoint_names={
            "mtp.0.main_norm.weight",
            "mtp.1.main_norm.weight",
        },
    )

    with pytest.raises(ValueError, match="mtp.1.main_norm.weight"):
        model.load_weights([("mtp.0.main_norm.weight", torch.ones(1))])


def test_dspark_ep1_rejects_only_the_missing_expert_source_component(
    monkeypatch,
) -> None:
    present = "mtp.0.ffn.experts.7.w1.weight"
    missing = "mtp.0.ffn.experts.7.w3.weight"
    model = _make_weight_loading_model(
        monkeypatch,
        expected_checkpoint_names={
            "mtp.0.main_norm.weight",
            present,
            missing,
        },
    )

    with pytest.raises(ValueError) as exc_info:
        model.load_weights(
            [
                ("mtp.0.main_norm.weight", torch.ones(1)),
                (present, torch.ones(1)),
            ]
        )

    assert present not in str(exc_info.value)
    assert missing in str(exc_info.value)


def test_dspark_skips_completeness_check_on_non_owning_pp_rank(monkeypatch) -> None:
    model = _make_weight_loading_model(
        monkeypatch,
        owns_layers=False,
        include_embed=True,
        shares_target_embed=False,
    )

    assert model.load_weights([]) == set()
