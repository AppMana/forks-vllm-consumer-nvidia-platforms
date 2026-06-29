# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.gpu.model_runner import _copy_or_reuse_pp_intermediate_tensor


def test_pp_intermediate_copy_uses_received_length() -> None:
    dst = torch.empty((4096, 8), dtype=torch.bfloat16)
    src = torch.arange(537 * 8, dtype=torch.float32).reshape(537, 8).to(torch.bfloat16)

    actual = _copy_or_reuse_pp_intermediate_tensor(dst, src, num_tokens=4096)

    assert actual.shape == src.shape
    torch.testing.assert_close(actual, src)


def test_pp_intermediate_copy_reuses_matching_view() -> None:
    src = torch.empty((537, 8), dtype=torch.bfloat16)

    actual = _copy_or_reuse_pp_intermediate_tensor(src, src, num_tokens=537)

    assert actual.shape == src.shape
    assert actual.data_ptr() == src.data_ptr()
