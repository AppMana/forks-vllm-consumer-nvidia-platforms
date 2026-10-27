# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native compact GGUF dense execution; the owning linear manages collectives."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.utils.b12x import B12xWarmupUnit


@dataclass(frozen=True)
class GGUFLinearSlice:
    source: Any
    rows: int
    columns: int
    row_start: int = 0
    column_start: int = 0


class B12xGGUFLinearMethod(LinearMethodBase):
    def __init__(
        self,
        slices=(),
        *,
        max_tokens,
        sources=(),
        partition=None,
        pad_to_block=False,
        retain_output_buffers=True,
    ):
        self.slices = tuple(slices)
        self.sources = tuple(sources)
        self.partition = partition
        self.pad_to_block = pad_to_block
        self.retain_output_buffers = retain_output_buffers
        if bool(self.slices) == bool(self.sources):
            raise ValueError("Supply either explicit GGUF slices or source matrices")
        if self.sources and partition not in ("column", "row", "replicated"):
            raise ValueError("Unknown native GGUF partition")
        self.max_tokens = int(max_tokens)
        if self.max_tokens <= 0:
            raise ValueError("Native GGUF linear needs sources and positive capacity")
        if self.slices and len({s.columns for s in self.slices}) != 1:
            raise ValueError("Merged native GGUF projections must share input width")
        self.weights: tuple[Any, ...] = ()
        self._states = []
        self._outputs = []
        self._output = None

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("Native GGUF linear requires BF16 activations")
        if self.sources:
            rank, size = layer.tp_rank, layer.tp_size
            if len(self.sources) != len(output_partition_sizes):
                raise ValueError("GGUF sources disagree with merged partition count")
            slices = []
            for source, rows in zip(self.sources, output_partition_sizes):
                if self.partition == "column":
                    expected_rows = source.rows
                    if self.pad_to_block:
                        expected_rows = (
                            ((source.rows + 256 * size - 1) // (256 * size))
                            * 256
                            * size
                        )
                    if source.columns != input_size or rows * size != expected_rows:
                        raise ValueError("GGUF column partition geometry disagrees")
                    slices.append(
                        GGUFLinearSlice(source, rows, input_size, row_start=rank * rows)
                    )
                elif self.partition == "row":
                    expected_columns = source.columns
                    if self.pad_to_block:
                        expected_columns = (
                            ((source.columns + 256 * size - 1) // (256 * size))
                            * 256
                            * size
                        )
                    if rows != source.rows or input_size != expected_columns:
                        raise ValueError("GGUF row partition geometry disagrees")
                    slices.append(
                        GGUFLinearSlice(
                            source,
                            rows,
                            input_size_per_partition,
                            column_start=rank * input_size_per_partition,
                        )
                    )
                else:
                    if (rows, input_size_per_partition) != (
                        source.rows,
                        source.columns,
                    ):
                        raise ValueError("GGUF replicated geometry disagrees")
                    slices.append(GGUFLinearSlice(source, rows, source.columns))
            self.slices = tuple(slices)
        if input_size_per_partition != self.slices[0].columns or tuple(
            output_partition_sizes
        ) != tuple(s.rows for s in self.slices):
            raise ValueError("Native GGUF slices disagree with linear partition")
        # Follow the constructor's default-device context without allocating
        # raw checkpoint Parameters. The GGUF loader supplies source descriptors.
        layer.register_buffer("_b12x_gguf_device", torch.empty(0), persistent=False)

    def process_weights_after_loading(self, layer):
        if getattr(layer, "is_bmm", False):
            raise NotImplementedError(
                "Grouped WO-A requires a grouped GGUF projection method"
            )
        self._prepare_slices(layer)

    def _prepare_slices(self, layer):
        from b12x.loader.gguf_dense import prepare_dense_partition

        if self.weights:
            raise RuntimeError("Reload requires rebuilding GGUF plans and captures")
        device = layer._b12x_gguf_device.device
        self.weights = tuple(
            prepare_dense_partition(
                s.source,
                device=device,
                row_start=s.row_start,
                rows=s.rows,
                column_start=s.column_start,
                columns=s.columns,
            )
            for s in self.slices
        )
        layer.b12x_warmup_provider = self
        # vLLM profiles before its general kernel warmup pass.
        self.warmup_launches(layer, token_counts=(self.max_tokens,))

    def get_b12x_warmup_unit(self, layer, token_counts, output_dtype):
        if output_dtype != torch.bfloat16:
            raise ValueError("Native GGUF linear warmup requires BF16")
        return B12xWarmupUnit(
            name="native GGUF linear",
            key=(type(self), id(self)),
            compile=lambda: self.warmup_launches(layer, token_counts=token_counts),
        )

    @torch.inference_mode()
    def warmup_launches(self, layer, *, token_counts):
        from b12x.gemm import blockscaled
        from b12x.preparation import PreparationSession, PreparedCall, require_prepared

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native GGUF linear planning must precede capture")
        counts = tuple(int(n) for n in token_counts)
        if not counts or min(counts) <= 0 or max(counts) > self.max_tokens:
            raise ValueError("Warmup exceeds native GGUF linear capacity")
        if self._states:
            return
        if not self.weights:
            raise RuntimeError("Load native GGUF linear weights before warmup")
        device = layer._b12x_gguf_device.device
        x = torch.zeros(
            self.max_tokens,
            self.slices[0].columns,
            dtype=torch.bfloat16,
            device=device,
        )
        with PreparationSession(
            device=device, autotune=False, compile_workers=1
        ) as session:
            for weight in self.weights:
                output = torch.empty(
                    self.max_tokens,
                    weight.out_features,
                    dtype=torch.bfloat16,
                    device=device,
                )
                query = blockscaled.query_from_call(x, weight, out=output)
                plan = blockscaled.plan(query)

                def factory(state, weight=weight, output=output):
                    return PreparedCall(
                        run=lambda: state.run(
                            x,
                            weight.values,
                            weight.metadata,
                            None,
                            out=output,
                        )
                    )

                session.prepare(
                    (
                        plan.request(
                            name="native GGUF linear",
                            prepare_call=factory,
                        ),
                    )
                )
                self._states.append(
                    require_prepared(plan, "gemm.blockscaled_precision")
                )
                self._outputs.append(output)
            session.freeze()
        self._output = (
            self._outputs[0]
            if len(self._outputs) == 1
            else torch.empty(
                self.max_tokens,
                sum(s.rows for s in self.slices),
                dtype=torch.bfloat16,
                device=device,
            )
        )
        if not self.retain_output_buffers:
            # Prepared kernels own packed weights and launch metadata, not the
            # output addresses. Let eager tensors follow their consumers' lifetimes;
            # CUDA capture retains its own static allocations in the graph pool.
            self._outputs.clear()
            self._output = None

    def _execution_outputs(self, m, device):
        if self.retain_output_buffers:
            assert self._output is not None
            return [out[:m] for out in self._outputs], self._output[:m]
        outputs = [
            torch.empty(m, s.rows, dtype=torch.bfloat16, device=device)
            for s in self.slices
        ]
        result = (
            outputs[0]
            if len(outputs) == 1
            else torch.empty(
                m, sum(s.rows for s in self.slices), dtype=torch.bfloat16, device=device
            )
        )
        return outputs, result

    def apply(self, layer, x, bias=None):
        if not self._states:
            raise RuntimeError("Native GGUF linear must be prepared before execution")
        if x.dtype != torch.bfloat16 or x.device != layer._b12x_gguf_device.device:
            raise ValueError("Native GGUF linear activation dtype/device mismatch")
        if x.ndim < 2 or x.shape[-1] != self.slices[0].columns or not x.is_contiguous():
            raise ValueError(
                "Native GGUF linear requires contiguous matching activations"
            )
        flat = x.view(-1, x.shape[-1])
        m = flat.shape[0]
        if m > self.max_tokens:
            raise ValueError("Native GGUF linear exceeds prepared capacity")
        outputs, result = self._execution_outputs(m, x.device)
        for state, weight, target in zip(self._states, self.weights, outputs):
            state.run(flat, weight.values, weight.metadata, None, out=target)
        if len(outputs) > 1:
            torch.cat(outputs, dim=-1, out=result)
        if bias is not None:
            result.add_(bias)
        return result.view(*x.shape[:-1], result.shape[-1])


class B12xGGUFGroupedLinearMethod(B12xGGUFLinearMethod):
    """Compact WO-A baseline using one prepared tensor-core GEMM per group.

    Group gathers and concatenation can retain capacity buffers or allocate
    live-row tensors. Both modes preserve the same prepared kernels and math.
    """

    def process_weights_after_loading(self, layer):
        if self.weights:
            raise RuntimeError("Reload requires rebuilding GGUF plans and captures")
        groups = getattr(layer, "bmm_batch_size", 0)
        if not getattr(layer, "is_bmm", False) or groups <= 0 or len(self.slices) != 1:
            raise ValueError("Native GGUF grouped method requires one grouped matrix")
        source = self.slices[0]
        if source.rows % groups or (source.rows // groups) % 8:
            raise ValueError("Native GGUF grouped rows must align to eight")
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
        self._group_inputs = (
            torch.empty(
                groups,
                self.max_tokens,
                source.columns,
                dtype=torch.bfloat16,
                device=layer._b12x_gguf_device.device,
            )
            if self.retain_output_buffers
            else None
        )

    def apply(self, layer, x, bias=None):
        raise RuntimeError("Grouped GGUF WO-A must use apply_grouped")

    def apply_grouped(self, layer, x):
        if not self._states or x.ndim != 3:
            raise ValueError("Native grouped GGUF input requires prepared [M,G,K]")
        if (
            x.dtype != torch.bfloat16
            or x.device != layer._b12x_gguf_device.device
            or x.shape[1:] != (len(self.slices), self.slices[0].columns)
        ):
            raise ValueError("Native grouped GGUF activation geometry/dtype mismatch")
        m = x.shape[0]
        if m > self.max_tokens:
            raise ValueError("Native grouped GGUF exceeds prepared capacity")
        outputs, result = self._execution_outputs(m, x.device)
        group_inputs = (
            self._group_inputs
            if self.retain_output_buffers
            else torch.empty(
                len(self.slices),
                m,
                self.slices[0].columns,
                dtype=torch.bfloat16,
                device=x.device,
            )
        )
        assert group_inputs is not None
        for group, (state, weight, target) in enumerate(
            zip(
                self._states,
                self.weights,
                outputs,
            )
        ):
            source = group_inputs[group, :m]
            source.copy_(x[:, group])
            state.run(source, weight.values, weight.metadata, None, out=target)
        if len(outputs) > 1:
            torch.cat(outputs, dim=-1, out=result)
        return result
