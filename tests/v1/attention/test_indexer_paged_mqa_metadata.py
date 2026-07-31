from vllm.v1.attention.backends.mla.indexer import (
    paged_mqa_logits_needs_deep_gemm_metadata,
)


def test_consumer_cuda_int8_skips_deep_gemm_metadata() -> None:
    assert not paged_mqa_logits_needs_deep_gemm_metadata(
        False, is_sm8x=True, is_sm12x=False
    )
    assert not paged_mqa_logits_needs_deep_gemm_metadata(
        False, is_sm8x=False, is_sm12x=True
    )


def test_fp4_and_datacenter_paths_keep_deep_gemm_metadata() -> None:
    assert paged_mqa_logits_needs_deep_gemm_metadata(True, is_sm8x=False, is_sm12x=True)
    assert paged_mqa_logits_needs_deep_gemm_metadata(
        False, is_sm8x=False, is_sm12x=False
    )
