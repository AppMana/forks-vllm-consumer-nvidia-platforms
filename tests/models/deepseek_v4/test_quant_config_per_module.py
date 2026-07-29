# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-module MoE quantization resolution from the checkpoint's own table.

``appmana/deepseek-v4-nvfp4-fp8`` lists ``layers.0..42.ffn.experts`` in
``quantization_config.quantized_layers`` as NVFP4 and leaves the MTP/DSpark
blocks on the base checkpoint's MXFP4 (e8m0 scales, group 32). Reading the
whole-model ``moe_quant_algo`` instead builds the draft's experts as NVFP4, and
nothing throws: ``_narrow_expert_data_for_padding`` narrows each parameter to
the loaded tensor and leaves the rest at ``torch.empty``, so the only trace is
a warning comparing uninitialised memory (a20abf7532).

The method constructors are stubbed: which class is chosen is the decision
under test, and constructing the real one drags in MoE backend validation that
has nothing to do with it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

NVFP4_LAYERS = range(43)
MTP_LAYER = 44


def make_quantization_config(*, with_table: bool = True) -> dict:
    config: dict = {
        "quant_method": "fp8",
        "moe_quant_algo": "NVFP4",
        "ignore": ["mtp.*"],
    }
    if with_table:
        # The exporter roots its keys at the transformer, not the wrapper.
        config["quantized_layers"] = {
            f"layers.{index}.ffn.experts": {"quant_algo": "NVFP4"}
            for index in NVFP4_LAYERS
        }
    return config


@pytest.fixture
def quant_config(monkeypatch):
    """A DeepseekV4FP8Config bound to a synthetic NVFP4-backbone checkpoint."""

    def _build(*, with_table: bool = True):
        import vllm.models.deepseek_v4.quant_config as quant_config_module

        hf_config = SimpleNamespace(
            expert_dtype="fp4",
            quantization_config=make_quantization_config(with_table=with_table),
        )
        monkeypatch.setattr(
            quant_config_module,
            "get_current_vllm_config",
            lambda: SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config)),
        )
        return quant_config_module.DeepseekV4FP8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
        )

    return _build


@pytest.fixture
def stub_moe_methods(monkeypatch):
    """Replace the two MoE method classes with distinguishable sentinels."""
    import vllm.models.deepseek_v4.quant_config as quant_config_module
    from vllm.model_executor.layers.quantization import modelopt

    class StubNvFp4Method:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class StubMxFp4Method:
        def __init__(self, *args, **kwargs):
            self.args = args

    monkeypatch.setattr(modelopt, "ModelOptNvFp4FusedMoE", StubNvFp4Method)
    monkeypatch.setattr(quant_config_module, "Mxfp4MoEMethod", StubMxFp4Method)
    return StubNvFp4Method, StubMxFp4Method


def routed_experts_layer():
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    layer = MagicMock(spec=RoutedExperts)
    layer.moe_config = MagicMock()
    return layer


def test_backbone_is_nvfp4_and_mtp_is_mxfp4(quant_config, stub_moe_methods) -> None:
    """The table decides per module, not one algorithm for the whole model.

    Would have caught a20abf7532: ``get_quant_method`` read the whole-model
    ``moe_quant_algo``, so the DSpark draft's MXFP4 experts were built as
    NVFP4 and half-filled from the checkpoint.
    """
    nvfp4_method, mxfp4_method = stub_moe_methods
    config = quant_config()
    layer = routed_experts_layer()

    backbone = config.get_quant_method(layer, "model.layers.5.ffn.experts")
    assert isinstance(backbone, nvfp4_method), type(backbone).__name__

    mtp = config.get_quant_method(layer, f"model.layers.{MTP_LAYER}.ffn.experts")
    assert isinstance(mtp, mxfp4_method), type(mtp).__name__


def test_resolved_algorithm_per_prefix(quant_config) -> None:
    """The resolution itself, without the method construction."""
    config = quant_config()

    assert config.moe_quant_algo == "NVFP4"
    assert config.moe_quant_algo_for("model.layers.0.ffn.experts") == "NVFP4"
    assert config.moe_quant_algo_for("model.layers.42.ffn.experts") == "NVFP4"
    # Absent from the table: not NVFP4, whatever the whole-model field says.
    assert config.moe_quant_algo_for(f"model.layers.{MTP_LAYER}.ffn.experts") != "NVFP4"

    layer = routed_experts_layer()
    assert config.is_mxfp4_quant("model.layers.5.ffn.experts", layer) is False
    assert config.is_mxfp4_quant(f"model.layers.{MTP_LAYER}.ffn.experts", layer) is True


def test_checkpoint_without_a_table_keeps_whole_model_behaviour(
    quant_config, stub_moe_methods
) -> None:
    """No ``quantized_layers`` table: every expert module follows
    ``moe_quant_algo``, exactly as before a20abf7532."""
    nvfp4_method, _ = stub_moe_methods
    config = quant_config(with_table=False)
    layer = routed_experts_layer()

    for prefix in (
        "model.layers.5.ffn.experts",
        f"model.layers.{MTP_LAYER}.ffn.experts",
    ):
        assert isinstance(config.get_quant_method(layer, prefix), nvfp4_method), prefix
