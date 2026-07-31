# SPDX-License-Identifier: Apache-2.0

from vllm.models.deepseek_v4.nvidia.model import _is_out_of_range_decoder_weight


def test_reduced_model_skips_only_out_of_range_decoder_weights() -> None:
    assert not _is_out_of_range_decoder_weight("layers.5.attn.attn_sink", 6)
    assert _is_out_of_range_decoder_weight("layers.6.attn.attn_sink", 6)
    assert _is_out_of_range_decoder_weight("model.layers.42.ffn.gate.weight", 6)
    assert not _is_out_of_range_decoder_weight("mtp.0.layers.0.ffn.weight", 6)
    assert not _is_out_of_range_decoder_weight("embed.weight", 6)
