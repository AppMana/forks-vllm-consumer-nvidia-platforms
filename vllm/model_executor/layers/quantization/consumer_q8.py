# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AOT Q8 inline decode and shared-workspace expansion for dense projections."""

import torch

from vllm.model_executor.layers.quantization.b12x_gguf_linear import (
    B12xGGUFLinearMethod,
    GGUFLinearSlice,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)


class ConsumerQ8LinearMethod(B12xGGUFLinearMethod):
    def __init__(self, *args, expand_min_tokens=512, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(expand_min_tokens, int) or expand_min_tokens <= 0:
            raise ValueError("Q8 expansion threshold must be a positive integer")
        self.expand_min_tokens = expand_min_tokens
        self._weight_workspace = None
        self._codes = None
        self._scales = None

    def _scratch(self):
        if self._codes is None:
            raise RuntimeError("Load Q8 weights before requesting scratch")
        if is_workspace_manager_initialized():
            return current_workspace_manager().get_simultaneous(
                (tuple(self._codes.shape), torch.bfloat16)
            )[0]
        if self._weight_workspace is None:
            raise RuntimeError("Q8 expansion workspace must be prepared")
        return self._weight_workspace

    def _prepare_slices(self, layer):
        if self._codes is not None:
            raise RuntimeError("Reload requires rebuilding Q8 captures")
        device = layer._b12x_gguf_device.device
        grouped = isinstance(self, ConsumerQ8GroupedLinearMethod)
        groups = len(self.slices) if grouped else 1
        n = self.slices[0].rows if grouped else sum(s.rows for s in self.slices)
        k = self.slices[0].columns
        if k % 64 or any(
            s.source.quant_type != "Q8_0" or s.columns != k or (grouped and s.rows != n)
            for s in self.slices
        ):
            raise ValueError("Consumer Q8 requires matching Q8 slices and K64")
        if device.type != "cuda":
            raise ValueError("Consumer Q8 requires CUDA weights")
        with torch.accelerator.device_index(device.index):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Q8 loading must precede capture")
            self._codes = torch.empty(groups, n, k, dtype=torch.int8, device=device)
            self._scales = torch.empty(
                groups, n, k // 32, dtype=torch.float16, device=device
            )
            offset = 0
            for index, source in enumerate(self.slices):
                group = index if grouped else 0
                for start in range(0, source.rows, 128):
                    count = min(128, source.rows - start)
                    raw = source.source.read_rows(
                        source.row_start + start,
                        count,
                        column_start=source.column_start,
                        columns=k,
                    )
                    codes, scales = self._pack_q8(raw.to(device).unsqueeze(0))
                    target = start if grouped else offset + start
                    self._codes[group, target : target + count].copy_(codes[0])
                    self._scales[group, target : target + count].copy_(scales[0])
                    del raw, codes, scales
                offset += source.rows
            self._finish_preparation()
            layer.b12x_warmup_provider = self
            self.warmup_launches(layer, token_counts=(self.max_tokens,))

    @staticmethod
    def _pack_q8(raw):
        from consumer_int8.gguf import pack_q8

        return pack_q8(raw)

    def _finish_preparation(self):
        assert self._codes is not None
        groups, n, _ = self._codes.shape
        self._output = torch.empty(
            self.max_tokens, groups, n, dtype=torch.bfloat16, device=self._codes.device
        )
        if not is_workspace_manager_initialized():
            self._weight_workspace = torch.empty_like(self._codes, dtype=torch.bfloat16)

    @torch.inference_mode()
    def warmup_launches(self, layer, *, token_counts):
        from consumer_int8.gguf import q8_expand_mm, q8_packed_mm

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Q8 warmup must precede capture")
        counts = tuple(token_counts)
        if not counts or min(counts) <= 0 or max(counts) > self.max_tokens:
            raise ValueError("Q8 warmup exceeds capacity")
        if self._codes is None:
            raise RuntimeError("Load Q8 weights before warmup")
        x = torch.zeros(
            self.max_tokens,
            self._codes.shape[0],
            self._codes.shape[2],
            device=self._codes.device,
            dtype=torch.bfloat16,
        )
        q8_packed_mm(x, self._codes, self._scales, self._output)
        q8_expand_mm(x, self._codes, self._scales, self._scratch(), self._output)

    def _run(self, x):
        from consumer_int8.gguf import q8_expand_mm, q8_packed_mm

        if self._output is None or self._codes is None:
            raise RuntimeError("Q8 execution requires prepared weights")
        if (
            x.ndim != 3
            or x.shape[1:] != (self._codes.shape[0], self._codes.shape[2])
            or x.dtype != torch.bfloat16
            or x.device != self._codes.device
        ):
            raise ValueError("Q8 activation geometry/dtype/device mismatch")
        m = x.shape[0]
        if m > self.max_tokens:
            raise ValueError("Q8 execution exceeds capacity")
        output = self._output[:m]
        if m >= self.expand_min_tokens:
            q8_expand_mm(x, self._codes, self._scales, self._scratch(), output)
        else:
            q8_packed_mm(x, self._codes, self._scales, output)
        return output

    def apply(self, layer, x, bias=None):
        if x.ndim < 2 or not x.is_contiguous():
            raise ValueError("Q8 linear requires contiguous activations")
        output = self._run(x.view(-1, 1, x.shape[-1])).flatten(1)
        if bias is not None:
            output.add_(bias)
        return output.view(*x.shape[:-1], output.shape[-1])


class ConsumerQ8GroupedLinearMethod(ConsumerQ8LinearMethod):
    def process_weights_after_loading(self, layer):
        groups = getattr(layer, "bmm_batch_size", 0)
        if not getattr(layer, "is_bmm", False) or groups <= 0 or len(self.slices) != 1:
            raise ValueError("Grouped Q8 requires one grouped matrix")
        source = self.slices[0]
        if source.rows % groups:
            raise ValueError("Grouped Q8 rows must divide evenly")
        rows = source.rows // groups
        self.slices = tuple(
            GGUFLinearSlice(
                source.source,
                rows,
                source.columns,
                row_start=source.row_start + group * rows,
                column_start=source.column_start,
            )
            for group in range(groups)
        )
        self._prepare_slices(layer)

    def apply(self, layer, x, bias=None):
        raise RuntimeError("Grouped Q8 must use apply_grouped")

    def apply_grouped(self, layer, x):
        return self._run(x).flatten(1)


class ConsumerQ8W8A8LinearMethod(ConsumerQ8LinearMethod):
    """Experimental channel requantization into native INT8 tensor MMA.

    This explicitly changes GGUF weight values and quantizes activations,
    including decode. It requires model quality and latency qualification.
    Only one INT8 weight copy survives loading; eager outputs own their
    lifetime and CUDA graph captures retain their own activation scratch.
    """

    @staticmethod
    def _pack_q8(raw):
        # W8A8 needs only the channel INT8 AOT library, not GGUF/BF16 kernels.
        if raw.dtype != torch.uint8 or raw.ndim != 4 or raw.shape[-1] != 34:
            raise ValueError("Expected Q8 uint8 [G,N,K/32,34] blocks")
        scales = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
        if not torch.isfinite(scales).all().item():
            raise ValueError("Q8 scales must be finite")
        codes = raw[..., 2:].contiguous().view(torch.int8)
        return codes.reshape(*raw.shape[:2], -1), scales

    def _finish_preparation(self):
        from consumer_int8.ops import (
            fused_channel_mm,
            quantize_grouped4096,
            quantize_rows,
        )

        assert self._codes is not None and self._scales is not None
        self._quantize_rows = quantize_rows
        self._quantize_grouped4096 = quantize_grouped4096
        self._fused_channel_mm = fused_channel_mm
        groups, n, k = self._codes.shape
        if n % 8 or k % 64:
            raise ValueError("Native W8A8 requires N8/K64 aligned weights")
        self._channel_scales = torch.empty(
            groups, n, dtype=torch.bfloat16, device=self._codes.device
        )
        # Bound conversion memory independently of the layer's full shape.
        for group in range(groups):
            for start in range(0, n, 128):
                codes = self._codes[group, start : start + 128]
                scales = self._scales[group, start : start + 128]
                dense = (
                    (codes.float().reshape(-1, k // 32, 32) * scales.float()[..., None])
                    .reshape(-1, k)
                    .bfloat16()
                )
                channel = (
                    (dense.float().abs().amax(1) / 127).clamp_min(1e-20).bfloat16()
                )
                converted = (
                    (dense.float() / channel.float()[:, None])
                    .round()
                    .clamp(-127, 127)
                    .to(torch.int8)
                )
                codes.copy_(converted)
                self._channel_scales[group, start : start + 128].copy_(channel)
        self._scales = None

    @torch.inference_mode()
    def warmup_launches(self, layer, *, token_counts):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("W8A8 warmup must precede capture")
        counts = tuple(token_counts)
        if not counts or min(counts) <= 0 or max(counts) > self.max_tokens:
            raise ValueError("W8A8 warmup exceeds capacity")
        if self._codes is None:
            raise RuntimeError("Load W8A8 weights before warmup")
        for m in counts:
            x = torch.zeros(
                m,
                self._codes.shape[0],
                self._codes.shape[2],
                device=self._codes.device,
                dtype=torch.bfloat16,
            )
            self._run(x)

    def _run(self, x):
        if self._codes is None:
            raise RuntimeError("W8A8 execution requires prepared weights")
        groups, n, k = self._codes.shape
        if (
            x.ndim != 3
            or x.shape[1:] != (groups, k)
            or x.dtype != torch.bfloat16
            or x.device != self._codes.device
            or x.shape[0] > self.max_tokens
        ):
            raise ValueError("W8A8 activation geometry/dtype/device/capacity mismatch")
        m = x.shape[0]
        output = torch.empty(groups, m, n, dtype=x.dtype, device=x.device)
        if m:
            grouped = groups > 1 and k == 4096 and x.is_contiguous()
            scratch_groups = groups if grouped else 1
            qa = torch.empty(scratch_groups, m, k, dtype=torch.int8, device=x.device)
            sa = torch.empty(scratch_groups, m, dtype=torch.float32, device=x.device)
            if grouped:
                self._quantize_grouped4096(x, qa, sa)
            for group in range(groups):
                slot = group if grouped else 0
                if not grouped:
                    values = x[:, group].contiguous()
                    self._quantize_rows(values, qa[slot], sa[slot])
                self._fused_channel_mm(
                    qa[slot],
                    self._codes[group],
                    sa[slot],
                    self._channel_scales[group],
                    output[group],
                    0,
                )
        return output.transpose(0, 1).contiguous()


class ConsumerQ8W8A8GroupedLinearMethod(
    ConsumerQ8W8A8LinearMethod, ConsumerQ8GroupedLinearMethod
):
    pass
