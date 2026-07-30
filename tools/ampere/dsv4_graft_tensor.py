# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replace one tensor in a safetensors shard with a verified donor tensor."""

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(-1).view(torch.uint8).numpy()
    return hashlib.sha256(raw).hexdigest()


def graft_tensor(
    base_path: Path,
    donor_path: Path,
    output_path: Path,
    tensor_name: str,
) -> tuple[str, str]:
    with safe_open(base_path, framework="pt", device="cpu") as base:
        tensors = {
            name: base.get_tensor(name)
            for name in base.keys()  # noqa: SIM118
        }

    if tensor_name not in tensors:
        raise KeyError(f"{tensor_name!r} is absent from {base_path}")

    with safe_open(donor_path, framework="pt", device="cpu") as donor:
        if tensor_name not in donor.keys():  # noqa: SIM118
            raise KeyError(f"{tensor_name!r} is absent from {donor_path}")
        replacement = donor.get_tensor(tensor_name)

    original = tensors[tensor_name]
    if replacement.shape != original.shape or replacement.dtype != original.dtype:
        raise ValueError(
            f"{tensor_name!r} mismatch: base={original.shape}/{original.dtype}, "
            f"donor={replacement.shape}/{replacement.dtype}"
        )

    original_digest = _digest(original)
    replacement_digest = _digest(replacement)
    tensors[tensor_name] = replacement
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_path)
    return original_digest, replacement_digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    args = parser.parse_args()

    original_digest, replacement_digest = graft_tensor(
        args.base.resolve(),
        args.donor.resolve(),
        args.output.resolve(),
        args.tensor,
    )
    print(f"base {args.tensor} sha256={original_digest}")
    print(f"donor {args.tensor} sha256={replacement_digest}")
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
