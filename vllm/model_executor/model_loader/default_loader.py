# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import glob
import os
import time
from collections.abc import Generator, Iterable
from typing import cast

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.torchao import torchao_version_at_least
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    get_quant_config,
    instanttensor_weights_iterator,
    maybe_download_from_modelscope,
    multi_thread_pt_weights_iterator,
    multi_thread_safetensors_weights_iterator,
    np_cache_weights_iterator,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.tracing import instrument
from vllm.transformers_utils.repo_utils import list_filtered_repo_files

logger = init_logger(__name__)


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: str | None
        """The optional model revision."""

        subfolder: str | None = None
        """The subfolder inside the model repo."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self.local_expert_ids: set[int] | None = None
        self.local_layer_range: tuple[int, int] | None = None
        self.is_first_pipeline_rank = True
        self.is_last_pipeline_rank = True

        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError(
                f"model_loader_extra_config must be a dict for load format "
                f"{load_config.load_format}, got {type(extra_config).__name__}"
            )
        allowed_keys = {
            "enable_multithread_load",
            "num_threads",
            "enable_weights_track",
        }
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

        enable_multithread_load = extra_config.get("enable_multithread_load", False)
        if not isinstance(enable_multithread_load, bool):
            raise ValueError(
                f"enable_multithread_load must be a bool, got "
                f"{type(enable_multithread_load).__name__}"
            )
        num_threads = extra_config.get("num_threads")
        if num_threads is not None and not (
            isinstance(num_threads, int) and num_threads > 0
        ):
            raise ValueError(
                f"num_threads must be a positive integer, got {num_threads!r}"
            )

        self.enable_weights_track: bool | None = extra_config.get(
            "enable_weights_track", None
        )

        # The multi-thread loader ignores safetensors_load_strategy, so reject
        # the combination instead of silently dropping the requested strategy.
        if extra_config.get("enable_multithread_load") and (
            load_config.safetensors_load_strategy not in (None, "lazy")
        ):
            raise ValueError(
                "enable_multithread_load does not support "
                "safetensors_load_strategy="
                f"{load_config.safetensors_load_strategy!r}; the multi-thread "
                "loader only implements the default lazy strategy."
            )

    def _prepare_weights(
        self,
        model_name_or_path: str,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = (
            maybe_download_from_modelscope(model_name_or_path, revision)
            or model_name_or_path
        )

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME

        # First check for 'auto' format that mistral files format are present.
        # This is to load mistral models with official format by default.
        if load_format == "auto":
            load_format = (
                "mistral"
                if len(
                    list_filtered_repo_files(
                        model_name_or_path=model_name_or_path,
                        allow_patterns=["consolidated*.safetensors"],
                        revision=revision,
                    )
                )
                > 0
                else "hf"
            )

        # Some quantized models use .pt files for storing the weights.
        if load_format == "hf":
            allow_patterns = ["*.safetensors", "*.bin"]
        elif (
            load_format == "safetensors"
            or load_format == "fastsafetensors"
            or load_format == "instanttensor"
        ):
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == "mistral":
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == "pt":
            allow_patterns = ["*.pt"]
        elif load_format == "npcache":
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        # Don't fall back to .pt for explicit safetensors formats; otherwise a
        # .pt file is matched and later opened as safetensors.
        if fall_back_to_pt and not use_safetensors:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                subfolder=subfolder,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        if subfolder is not None:
            hf_folder = os.path.join(hf_folder, subfolder)

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            matched_files = glob.glob(os.path.join(hf_folder, pattern))
            hf_weights_files.extend(
                path for path in matched_files if path not in hf_weights_files
            )
            if matched_files:
                if pattern.endswith(".safetensors"):
                    use_safetensors = True
                if allow_patterns_overrides is None:
                    break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local and len(hf_weights_files) > 1:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    cache_dir=self.load_config.download_dir,
                    subfolder=subfolder,
                    revision=revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files,
                hf_folder,
                index_file,
                # An explicit allow_patterns override selects a deliberate
                # subset of the checkpoint; the index completeness check
                # must not demand the excluded shards.
                allow_partial=allow_patterns_overrides is not None,
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        extra_config = self.load_config.model_loader_extra_config
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if self.load_config.load_format == "npcache":
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
        elif use_safetensors:
            if self.load_config.load_format == "fastsafetensors":
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            elif self.load_config.load_format == "instanttensor":
                weights_iterator = instanttensor_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            else:
                if extra_config.get("enable_multithread_load"):
                    weights_iterator = multi_thread_safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        max_workers=extra_config.get(
                            "num_threads", self.DEFAULT_NUM_THREADS
                        ),
                    )
                else:
                    weights_iterator = safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        self.load_config.safetensors_load_strategy,
                        local_expert_ids=self.local_expert_ids,
                        local_layer_range=self.local_layer_range,
                        is_first_pipeline_rank=self.is_first_pipeline_rank,
                        is_last_pipeline_rank=self.is_last_pipeline_rank,
                        safetensors_prefetch_num_threads=(
                            self.load_config.safetensors_prefetch_num_threads
                        ),
                        safetensors_prefetch_block_size=(
                            self.load_config.safetensors_prefetch_block_size
                        ),
                    )
        else:
            if extra_config.get("enable_multithread_load"):
                weights_iterator = multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                )
            else:
                weights_iterator = pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_name_or_path=model_config.model,
            subfolder=None,
            revision=model_config.revision,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )

    def _init_ep_weight_filter(self, model_config: ModelConfig) -> None:
        """Compute local expert ids for EP weight filtering.

        When expert parallelism is active, each rank only needs a subset of
        expert weights.  By computing the set upfront we can skip non-local
        expert tensors *before* reading them from disk.
        """
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config

        if not (
            model_config.is_moe
            and parallel_config.enable_expert_parallel
            and parallel_config.enable_ep_weight_filter
        ):
            return

        # When EPLB is enabled, redundant physical expert slots may map to
        # logical experts that belong to other ranks in the default partition.
        # The weight loader needs to see ALL logical expert weights so it can
        # populate these redundant slots.  Skip the filter entirely.
        if parallel_config.enable_eplb:
            return

        num_experts = model_config.get_num_experts()
        if num_experts <= 0:
            return

        # EP size/rank computation mirrors FusedMoEParallelConfig.make():
        #   ep_size = dp_size * pcp_size * tp_size (flattened)
        #   ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
        from vllm.distributed import (
            get_dp_group,
            get_pcp_group,
            get_tensor_model_parallel_rank,
        )

        dp_size = parallel_config.data_parallel_size
        tp_size = parallel_config.tensor_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        ep_size = dp_size * pcp_size * tp_size
        ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank

        self.local_expert_ids = compute_local_expert_ids(
            num_experts,
            ep_size,
            ep_rank,
            placement=parallel_config.expert_placement_strategy,
        )
        if self.local_expert_ids is not None:
            logger.info_once(
                "EP weight filter: ep_size=%d, ep_rank=%d, loading %d/%d experts",
                ep_size,
                ep_rank,
                len(self.local_expert_ids),
                num_experts,
            )

    def _init_pp_weight_filter(self, model_config: ModelConfig) -> None:
        """Compute this rank's local hidden-layer range for PP weight
        filtering.

        Each PP rank only holds a contiguous slice of the model's hidden
        layers (vllm.model_executor.models.utils.make_layers). Without this,
        safetensors_weights_iterator reads every layer's weights from every
        shard on every rank -- for a pp_size-way split that means each rank
        downloads close to the *entire* checkpoint instead of its own
        1/pp_size share. Computing the range upfront and passing it to the
        iterator lets us skip non-local per-layer tensors *before* reading
        them from disk, mirroring _init_ep_weight_filter's expert-level
        filtering.
        """
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config

        if parallel_config.pipeline_parallel_size <= 1:
            return

        start, end = model_config.get_layers_start_end_indices(parallel_config)
        self.local_layer_range = (start, end)
        from vllm.distributed import get_pp_group

        pp_group = get_pp_group()
        self.is_first_pipeline_rank = pp_group.is_first_rank
        self.is_last_pipeline_rank = pp_group.is_last_rank
        logger.info_once(
            "PP weight filter: pp_size=%d, loading layers [%d, %d) of %d",
            parallel_config.pipeline_parallel_size,
            start,
            end,
            model_config.get_total_num_hidden_layers(),
        )

    @instrument(span_name="Load weights")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        if model_config.quantization == "torchao":
            quant_config = get_quant_config(model_config, self.load_config)
            if (
                hasattr(quant_config, "is_checkpoint_torchao_serialized")
                and quant_config.is_checkpoint_torchao_serialized
                and torchao_version_at_least("0.15.0")
            ):
                self.load_config.safetensors_load_strategy = "torchao"

        self._init_ep_weight_filter(model_config)
        self._init_pp_weight_filter(model_config)

        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))

        self.counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
        if loaded_weights is None or self.enable_weights_track is False:
            # No tracking data, or the audit was explicitly turned off.
            return
        # A missing parameter is only *provably* a bug for a non-quantized
        # model, where every parameter must come from the checkpoint; that
        # case stays fatal. Quantized models are audited too -- they used to
        # be skipped entirely, which is how a silently uninitialized scale
        # survived -- but report rather than raise, because quantization
        # methods legitimately derive some parameters after loading. The one
        # exception, fatal in both modes, is a parameter a quantization
        # method has declared it cannot derive (see
        # QuantizeMethodBase.checkpoint_required_params).
        strict = bool(self.enable_weights_track) or model_config.quantization is None
        self.track_weights_loading(model, loaded_weights, strict=strict)

    def track_weights_loading(
        self,
        model: nn.Module,
        loaded_weights: set[str] | None,
        strict: bool = True,
    ) -> None:
        if loaded_weights is None:
            return
        weights_to_load = {name for name, _ in model.named_parameters()}
        # Parameters that may legitimately be absent from the checkpoint:
        # kv-cache scales and online-quantization scales, plus anything a
        # module with a process_weights_after_loading hook may fill in later.
        derivable: set[str] = set()
        # ... except what the quantization method itself declares it must
        # read from the checkpoint. Those are collected across every module
        # first, so an ancestor module's blanket whitelist cannot re-hide a
        # descendant's required parameter.
        required_not_loaded: set[str] = set()
        for name, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None:
                continue
            has_online_quant = getattr(quant_method, "uses_meta_device", False)
            has_postprocess_quant = getattr(
                quant_method, "process_weights_after_loading", None
            )
            if not (has_online_quant or has_postprocess_quant):
                continue
            get_required = getattr(quant_method, "checkpoint_required_params", None)
            required = get_required(module) if get_required is not None else frozenset()
            for param_name, _ in module.named_parameters():
                full_name = f"{name}.{param_name}" if name else param_name
                if param_name in required and full_name not in loaded_weights:
                    required_not_loaded.add(full_name)
                else:
                    derivable.add(full_name)
        derivable -= required_not_loaded

        if required_not_loaded:
            raise ValueError(
                "Following weights were not initialized from checkpoint and "
                "cannot be derived by the quantization method, so they would "
                "be read as uninitialized memory: "
                f"{sorted(required_not_loaded)}"
            )

        weights_not_loaded = weights_to_load - loaded_weights - derivable
        if not weights_not_loaded:
            return
        if strict:
            raise ValueError(
                "Following weights were not initialized from "
                f"checkpoint: {weights_not_loaded}"
            )
        listed = sorted(weights_not_loaded)
        logger.warning(
            "%d weights were not initialized from the checkpoint: %s%s. The "
            "model is quantized, so some of these may be filled in by "
            "process_weights_after_loading; any that are not are "
            "uninitialized memory.",
            len(listed),
            listed[:32],
            "" if len(listed) <= 32 else f" (+{len(listed) - 32} more)",
        )
