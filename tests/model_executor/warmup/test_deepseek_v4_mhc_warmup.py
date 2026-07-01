# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types

import torch

from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    _find_first_mhc_layer,
    _warmup_layer_mhc,
)


class FakeMHCLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 8
        self.hc_mult = 2
        self.rms_norm_eps = 1e-6
        self.hc_eps = 1e-6
        self.hc_post_alpha = 1.0
        self.hc_sinkhorn_iters = 2
        self.hc_attn_fn = torch.empty(8, 16, dtype=torch.float32)
        self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_attn_base = torch.empty(8, dtype=torch.float32)
        self.hc_ffn_fn = torch.empty(8, 16, dtype=torch.float32)
        self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_ffn_base = torch.empty(8, dtype=torch.float32)
        self.pre_calls: list[tuple[int, torch.Tensor]] = []
        self.fused_calls: list[tuple[int, torch.Tensor, int]] = []

    def hc_pre(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del scale, base
        self.pre_calls.append((residual.shape[0], fn))
        post_mix = torch.empty(residual.shape[0], self.hc_mult, 1, dtype=torch.float32)
        comb_mix = torch.empty(
            residual.shape[0], self.hc_mult, self.hc_mult, dtype=torch.float32
        )
        layer_input = torch.empty(
            residual.shape[0], self.hidden_size, dtype=residual.dtype
        )
        return layer_input, post_mix, comb_mix

    def mhc_fused_post_pre(
        self,
        layer_input: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_alpha: float,
        hc_sinkhorn_iters: int,
        n_splits: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del layer_input, post_mix, comb_mix, scale, base
        assert rms_norm_eps == self.rms_norm_eps
        assert hc_eps == self.hc_eps
        assert hc_sinkhorn_eps == self.hc_eps
        assert hc_post_alpha == self.hc_post_alpha
        assert hc_sinkhorn_iters == self.hc_sinkhorn_iters
        self.fused_calls.append((residual.shape[0], fn, n_splits))
        return (
            torch.empty_like(residual),
            torch.empty(residual.shape[0], self.hc_mult, 1, dtype=torch.float32),
            torch.empty(
                residual.shape[0], self.hc_mult, self.hc_mult, dtype=torch.float32
            ),
            torch.empty(residual.shape[0], self.hidden_size, dtype=residual.dtype),
        )


def test_mhc_warmup_exercises_fused_post_pre() -> None:
    layer = FakeMHCLayer()

    _warmup_layer_mhc(layer, [1, 2, 4])

    expected_sizes = [1, 1, 2, 2, 4, 4]
    assert [size for size, _ in layer.pre_calls] == expected_sizes
    assert [size for size, _, _ in layer.fused_calls] == expected_sizes
    assert [n_splits for _, _, n_splits in layer.fused_calls] == [1] * 6
    expected_fns = [
        layer.hc_attn_fn,
        layer.hc_ffn_fn,
        layer.hc_attn_fn,
        layer.hc_ffn_fn,
        layer.hc_attn_fn,
        layer.hc_ffn_fn,
    ]
    assert all(
        actual is expected
        for actual, expected in zip(
            [fn for _, fn, _ in layer.fused_calls], expected_fns
        )
    )


def test_mhc_layer_discovery_matches_live_deepseek_v4_layer_shape() -> None:
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(model_type="deepseek_v4")
    model.layer = FakeMHCLayer()

    assert _find_first_mhc_layer(model) is model.layer
