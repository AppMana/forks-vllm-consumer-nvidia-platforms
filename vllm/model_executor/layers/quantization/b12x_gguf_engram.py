# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP head-sharded Engram lookup from the original disk-resident Q5_K table."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class GGUFEngramSource:
    path: Path
    name: str
    offset: int
    rows: int
    columns: int


class B12xGGUFEngramEmbedding(torch.nn.Module):
    def __init__(
        self, source, layout, layer_hash_index, *, max_tokens, tp_rank, tp_size, device
    ):
        super().__init__()
        from b12x.sequence.embedding._disk import Q5KDiskTable

        if source.columns != 256 or layout.head_dim != 256:
            raise ValueError("GGUF Engram requires 256-wide Q5_K rows")
        if source.rows != sum(layout.num_embeddings):
            raise ValueError(
                "GGUF Engram table differs from configured layer row counts"
            )
        if not 0 <= layer_hash_index < len(layout.num_embeddings):
            raise ValueError("Invalid Engram layer table index")
        if not 0 <= tp_rank < tp_size or max_tokens <= 0:
            raise ValueError("Invalid GGUF Engram TP rank or token capacity")
        head_sizes = tuple(
            size for order in layout.primes[layer_hash_index] for size in order
        )
        if (
            not head_sizes
            or any(size <= 0 for size in head_sizes)
            or len(head_sizes) % tp_size
        ):
            raise ValueError(
                "GGUF Engram requires positive heads evenly split across TP"
            )
        self.gguf_source = source
        self.dim = source.columns
        self.num_embeddings = layout.num_embeddings[layer_hash_index]
        if sum(head_sizes) > self.num_embeddings:
            raise ValueError("Engram hash heads exceed layer table rows")
        self.n_hash_cols = len(head_sizes)
        self.tp_size, self.dp_size = tp_size, 1
        self.cpu_offload = True
        self.dp_shared_memory = False
        self.part_n_hash_cols = self.n_hash_cols // tp_size
        self.head_start = tp_rank * self.part_n_hash_cols
        self.vocab_start_idx = sum(head_sizes[: self.head_start])
        self.vocab_end_idx = sum(head_sizes[: self.head_start + self.part_n_hash_cols])
        self.row_start = sum(layout.num_embeddings[:layer_hash_index])
        self.max_tokens = max_tokens
        self.register_buffer(
            "_ids",
            torch.empty(
                max_tokens, self.part_n_hash_cols, dtype=torch.int64, device=device
            ),
            persistent=False,
        )
        self.table = Q5KDiskTable(
            path=source.path,
            offset=source.offset,
            table_rows=source.rows,
            max_lookups=max_tokens * self.part_n_hash_cols,
            device=device,
            shard_start=self.row_start + self.vocab_start_idx,
            shard_end=self.row_start + self.vocab_end_idx,
        )

    def lookup(self, indices, out, background=False):
        del background
        count = indices.shape[0]
        if (
            indices.ndim != 2
            or indices.shape[1] != self.n_hash_cols
            or indices.dtype not in (torch.int32, torch.int64)
            or indices.device != self._ids.device
            or count > self.max_tokens
        ):
            raise ValueError("GGUF Engram hashes differ from prepared geometry")
        if out.shape != (count, self.part_n_hash_cols, self.dim):
            raise ValueError("GGUF Engram requires local-head BF16 output")
        # Called at the existing eager prefetch boundary, outside graph capture.
        local = indices[:, self.head_start : self.head_start + self.part_n_hash_cols]
        ids = self._ids[:count]
        torch.add(local, self.row_start, out=ids)
        self.table.lookup(ids, out)

    def close(self):
        self.table.close()
