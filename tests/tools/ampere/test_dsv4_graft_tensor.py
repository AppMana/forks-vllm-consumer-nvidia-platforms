# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.ampere.dsv4_graft_tensor import graft_tensor


def test_graft_tensor_replaces_only_selected_tensor(tmp_path):
    base_path = tmp_path / "base.safetensors"
    donor_path = tmp_path / "donor.safetensors"
    output_path = tmp_path / "output.safetensors"
    base = {
        "head.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        "norm.weight": torch.arange(4, dtype=torch.bfloat16),
    }
    donor = {
        "head.weight": torch.full((2, 4), 42, dtype=torch.bfloat16),
        "norm.weight": torch.full((4,), 99, dtype=torch.bfloat16),
    }
    save_file(base, base_path)
    save_file(donor, donor_path)

    graft_tensor(base_path, donor_path, output_path, "head.weight")

    with safe_open(output_path, framework="pt", device="cpu") as output:
        assert torch.equal(output.get_tensor("head.weight"), donor["head.weight"])
        assert torch.equal(output.get_tensor("norm.weight"), base["norm.weight"])
