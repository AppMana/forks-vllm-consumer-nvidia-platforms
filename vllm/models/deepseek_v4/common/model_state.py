# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.models.deepseek_v41.nvidia.model_state import _pad_replayed_slots_kernel
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup


class DeepseekV4ModelState(DefaultModelState):
    """DefaultModelState plus prefix replay for the circular C4 compressor state.

    The C4 compressor and indexer-compressor rings are not prefix cached, and
    the first C4 group after a hit also compresses the group before it. The
    scheduler therefore recomputes the hit's last ``prefix_replay_tokens``
    tokens (``NewRequestData.replay_start``); here their slots in the
    prefix-cacheable groups are padded, so the cached sliding-window and
    compressed KV stay as they are while the rings are refilled. Attention
    still reads the full cached context: unlike DeepSeek V4.1 bounded replay,
    the sliding-window KV below the replay start is cached and valid.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        # Per request state index; batches gather from it (see prepare_attn).
        self._replay_start_np = np.zeros(self.max_num_reqs, dtype=np.int32)
        self._replay_start = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self._replay_start_staging = UvaBufferPool(self.max_num_reqs, torch.int32)
        # The replay window and the indices of the prefix-cacheable groups, from
        # the KV cache config on first use; None until then.
        self._replay: tuple[int, torch.Tensor] | None = None

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self._replay_start_np[req_index] = new_req_data.replay_start

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        if self._replay is None:
            specs = [group.kv_cache_spec for group in kv_cache_config.kv_cache_groups]
            self._replay = (
                max((spec.prefix_replay_tokens for spec in specs), default=0),
                torch.tensor(
                    [i for i, spec in enumerate(specs) if spec.prefix_cacheable],
                    dtype=torch.int32,
                    device=self.device,
                ),
            )
        window, cacheable_groups = self._replay
        num_reqs = input_batch.num_reqs
        if window and cacheable_groups.numel() and num_reqs:
            # Decode rows sit above the hit, so only prefills carry a replay
            # start; dummy batches (captures, profiling) carry none.
            replay_start_np = np.where(
                input_batch.is_prefilling_np[:num_reqs],
                self._replay_start_np[input_batch.idx_mapping_np[:num_reqs]],
                0,
            ).astype(np.int32)
            if replay_start_np.any():
                replay_start = self._replay_start_staging.copy_to_gpu(
                    replay_start_np, out=self._replay_start[:num_reqs]
                )
                _pad_replayed_slots_kernel[(num_reqs,)](
                    slot_mappings,
                    slot_mappings.stride(0),
                    cacheable_groups,
                    input_batch.query_start_loc,
                    input_batch.positions,
                    replay_start,
                    window,
                    PAD_SLOT_ID,
                    NUM_GROUPS=cacheable_groups.numel(),
                    BLOCK=1024,
                )
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
