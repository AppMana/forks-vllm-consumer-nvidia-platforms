# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Q6 prefill expansion sharing b12x's compact weights and worker scratch."""

import torch

from vllm.model_executor.layers.quantization.b12x_gguf_linear import (
    B12xGGUFGroupedLinearMethod,
    B12xGGUFLinearMethod,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)


class ConsumerQ6LinearMethod(B12xGGUFLinearMethod):
    def __init__(self, *args, expand_min_tokens=512, **kwargs):
        super().__init__(*args, **kwargs)
        if type(expand_min_tokens) is not int or expand_min_tokens <= 0:
            raise ValueError("Q6 expansion threshold must be a positive integer")
        self.expand_min_tokens = expand_min_tokens
        self._weight_workspace = None
        self._expanded_counts = set()

    def _scratch(self):
        shape = (sum(s.rows for s in self.slices), self.slices[0].columns)
        if is_workspace_manager_initialized():
            return current_workspace_manager().get_simultaneous(
                (shape, torch.bfloat16)
            )[0]
        if self._weight_workspace is None:
            raise RuntimeError("Q6 expansion workspace must be prepared")
        return self._weight_workspace

    @torch.inference_mode()
    def warmup_launches(self, layer, *, token_counts):
        counts = tuple(int(n) for n in token_counts)
        if not counts or min(counts) <= 0 or max(counts) > self.max_tokens:
            raise ValueError("Warmup exceeds native GGUF linear capacity")
        if any(s.source.quant_type != "Q6_K" for s in self.slices):
            raise ValueError("Consumer Q6 requires only Q6_K slices")
        device = layer._b12x_gguf_device.device
        if torch.cuda.get_device_capability(device)[0] >= 12:
            super().warmup_launches(layer, token_counts=counts)
            if self._output is None:
                raise RuntimeError("Q6 warmup requires prepared output storage")
            if self.max_tokens < self.expand_min_tokens:
                return
            min_expand = self.expand_min_tokens
            extra_counts = {self.max_tokens, self.expand_min_tokens}
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Native GGUF linear planning must precede capture")
            if not self.weights:
                raise RuntimeError("Load native GGUF linear weights before warmup")
            if self._output is None:
                self._output = torch.empty(
                    self.max_tokens,
                    sum(s.rows for s in self.slices),
                    dtype=torch.bfloat16,
                    device=device,
                )
            min_expand = 1
            extra_counts = {self.max_tokens}
        if not is_workspace_manager_initialized() and self._weight_workspace is None:
            self._weight_workspace = torch.empty(
                sum(s.rows for s in self.slices),
                self.slices[0].columns,
                dtype=torch.bfloat16,
                device=self._output.device,
            )
        expansion_counts = sorted((set(counts) | extra_counts) - self._expanded_counts)
        grouped = isinstance(self, B12xGGUFGroupedLinearMethod)
        groups = len(self.slices) if grouped else 1
        for count in expansion_counts:
            if count < min_expand:
                continue
            x = torch.zeros(
                count,
                groups,
                self.slices[0].columns,
                dtype=torch.bfloat16,
                device=self._output.device,
            )
            self._expanded(x if grouped else x[:, 0], grouped=grouped)
            self._expanded_counts.add(count)

    def _expanded(self, x, *, grouped):
        from consumer_int8.gguf import q6_packed_expand

        if self._output is None:
            raise RuntimeError("Q6 expansion requires prepared output storage")
        workspace = self._scratch()
        offset = 0
        for weight in self.weights:
            end = offset + weight.out_features
            q6_packed_expand(weight.values, weight.metadata, workspace[offset:end])
            offset = end
        output = self._output[: x.shape[0]]
        if grouped:
            groups = len(self.weights)
            n, k = self.slices[0].rows, self.slices[0].columns
            torch.bmm(
                x.transpose(0, 1),
                workspace.view(groups, n, k).transpose(1, 2),
                out=output.view(x.shape[0], groups, n).transpose(0, 1),
            )
        else:
            torch.bmm(
                x.unsqueeze(0), workspace.t().unsqueeze(0), out=output.unsqueeze(0)
            )
        return output

    def apply(self, layer, x, bias=None):
        if x.ndim < 2 or x.shape[-1] != self.slices[0].columns:
            raise ValueError("Q6 linear activation geometry mismatch")
        m = x.numel() // x.shape[-1]
        if self._states and m < self.expand_min_tokens:
            return super().apply(layer, x, bias)
        if (
            self._output is None
            or m > self.max_tokens
            or not x.is_contiguous()
            or x.dtype != torch.bfloat16
            or x.device != self._output.device
        ):
            raise ValueError(
                "Q6 linear requires prepared matching BF16 activations within capacity"
            )
        result = self._expanded(x.view(m, x.shape[-1]), grouped=False)
        if bias is not None:
            result.add_(bias)
        return result.view(*x.shape[:-1], result.shape[-1])


class ConsumerQ6GroupedLinearMethod(
    ConsumerQ6LinearMethod, B12xGGUFGroupedLinearMethod
):
    def apply(self, layer, x, bias=None):
        raise RuntimeError("Grouped Q6 must use apply_grouped")

    def apply_grouped(self, layer, x):
        if x.ndim != 3:
            raise ValueError("Q6 grouped input must have shape [M,G,K]")
        if self._states and x.shape[0] < self.expand_min_tokens:
            return super().apply_grouped(layer, x)
        if (
            self._output is None
            or x.shape[0] > self.max_tokens
            or x.shape[1:] != (len(self.slices), self.slices[0].columns)
            or x.dtype != torch.bfloat16
            or x.device != self._output.device
            or any(s.rows != self.slices[0].rows for s in self.slices)
        ):
            raise ValueError("Q6 grouped activation geometry/dtype/capacity mismatch")
        return self._expanded(x, grouped=True)
