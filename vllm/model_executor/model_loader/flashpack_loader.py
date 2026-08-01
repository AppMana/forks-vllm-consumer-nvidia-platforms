# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manifest-driven, sharded FlashPack model loading.

This loader deliberately does not use FlashPack's ``assign_from_file`` API.
vLLM models, especially quantized models, rely on their custom
``load_weights`` implementations to fuse and transform checkpoint tensors.
Instead, each selected FlashPack part is exposed as the original
``(name, tensor)`` stream consumed by ``model.load_weights``.

The manifest is read before any part is opened. Pipeline-local parts are
selected from its weight map, then each part's footer and checksum are
validated before its payload is transferred. Only one part is kept alive at
a time.

Each manifest part declares ``sha256``, ``size``, ``kind`` (``model`` or
``mtp``), and ``pipeline_stage`` (``any``, ``first``, or ``last``). The
stage annotation keeps embedding-only and head-only shards off pipeline ranks
that cannot consume them; numbered layer tensors are additionally selected by
the normal vLLM PP layer-range filter.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight
from vllm.model_executor.model_loader.pp_weight_filter import (
    should_skip_pp_weight,
)
from vllm.transformers_utils.repo_utils import hf_api

logger = init_logger(__name__)

FLASHPACK_INDEX_NAME = "model.flashpack.index.json"
FLASHPACK_INDEX_FORMAT = "vllm_sharded_flashpack_v1"


@dataclass(frozen=True)
class FlashPackPart:
    filename: str
    sha256: str
    size: int
    kind: str
    pipeline_stage: str


@dataclass(frozen=True)
class FlashPackIndex:
    source_repo_id: str
    source_revision: str
    weight_map: dict[str, str]
    parts: tuple[FlashPackPart, ...]

    @property
    def part_map(self) -> dict[str, FlashPackPart]:
        return {part.filename: part for part in self.parts}


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"FlashPack index field {field!r} must be a non-empty string")
    return value


def _validate_part_filename(filename: str) -> None:
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts or filename.endswith("/"):
        raise ValueError(f"Unsafe FlashPack part filename: {filename!r}")
    if path.suffix != ".flashpack":
        raise ValueError(f"FlashPack part must end in .flashpack: {filename!r}")


def parse_flashpack_index(data: Any) -> FlashPackIndex:
    """Validate and parse ``model.flashpack.index.json``.

    The index is intentionally strict. A malformed or incomplete manifest
    must not silently fall back to another checkpoint representation.
    """
    if not isinstance(data, dict):
        raise ValueError("FlashPack index must be a JSON object")
    if data.get("format") != FLASHPACK_INDEX_FORMAT:
        raise ValueError(
            "Unsupported FlashPack index format: "
            f"{data.get('format')!r}; expected {FLASHPACK_INDEX_FORMAT!r}"
        )

    source = data.get("source")
    if not isinstance(source, dict):
        raise ValueError("FlashPack index must contain a source object")
    source_repo_id = _require_nonempty_string(source.get("repo_id"), "source.repo_id")
    source_revision = _require_nonempty_string(
        source.get("revision"), "source.revision"
    )

    raw_parts = data.get("parts")
    if not isinstance(raw_parts, dict) or not raw_parts:
        raise ValueError("FlashPack index parts must be a non-empty object")

    parts: list[FlashPackPart] = []
    for filename, raw_part in raw_parts.items():
        filename = _require_nonempty_string(filename, "parts filename")
        _validate_part_filename(filename)
        if not isinstance(raw_part, dict):
            raise ValueError(f"FlashPack part {filename!r} must be an object")
        sha256 = _require_nonempty_string(
            raw_part.get("sha256"), f"parts.{filename}.sha256"
        ).lower()
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError(f"Invalid SHA-256 for FlashPack part {filename!r}")
        size = raw_part.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"Invalid size for FlashPack part {filename!r}")
        kind = raw_part.get("kind", "model")
        if kind not in {"model", "mtp"}:
            raise ValueError(f"Invalid kind for FlashPack part {filename!r}: {kind!r}")
        pipeline_stage = raw_part.get("pipeline_stage", "any")
        if pipeline_stage not in {"any", "first", "last"}:
            raise ValueError(
                f"Invalid pipeline_stage for FlashPack part {filename!r}: "
                f"{pipeline_stage!r}"
            )
        parts.append(FlashPackPart(filename, sha256, size, kind, pipeline_stage))

    raw_weight_map = data.get("weight_map")
    if not isinstance(raw_weight_map, dict) or not raw_weight_map:
        raise ValueError("FlashPack index weight_map must be a non-empty object")
    part_names = {part.filename for part in parts}
    weight_map: dict[str, str] = {}
    for name, filename in raw_weight_map.items():
        name = _require_nonempty_string(name, "weight_map tensor name")
        filename = _require_nonempty_string(
            filename, f"weight_map.{name} part filename"
        )
        if filename not in part_names:
            raise ValueError(
                f"Tensor {name!r} references undeclared FlashPack part {filename!r}"
            )
        weight_map[name] = filename

    unreferenced = part_names - set(weight_map.values())
    if unreferenced:
        raise ValueError(
            f"FlashPack index declares parts with no tensors: {sorted(unreferenced)}"
        )

    return FlashPackIndex(
        source_repo_id=source_repo_id,
        source_revision=source_revision,
        weight_map=weight_map,
        parts=tuple(parts),
    )


def select_flashpack_parts(
    index: FlashPackIndex,
    should_load_weight: Callable[[str], bool],
    *,
    include_mtp: bool,
    is_first_pipeline_stage: bool = True,
    is_last_pipeline_stage: bool = True,
) -> tuple[FlashPackPart, ...]:
    """Select parts with at least one local tensor without opening payloads."""
    needed_filenames = {
        filename
        for name, filename in index.weight_map.items()
        if should_load_weight(name)
    }
    return tuple(
        part
        for part in index.parts
        if part.filename in needed_filenames
        and (include_mtp or part.kind != "mtp")
        and (part.pipeline_stage != "first" or is_first_pipeline_stage)
        and (part.pipeline_stage != "last" or is_last_pipeline_stage)
    )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_part_file(path: str, part: FlashPackPart) -> None:
    try:
        size = os.path.getsize(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing FlashPack part {part.filename!r}") from exc
    if size != part.size:
        raise ValueError(
            f"FlashPack part {part.filename!r} has size {size}, expected {part.size}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != part.sha256:
        raise ValueError(
            f"FlashPack part {part.filename!r} failed SHA-256 validation: "
            f"got {actual_sha256}, expected {part.sha256}"
        )


def _validate_footer_tensor_names(
    metadata: Any,
    expected_names: set[str],
    filename: str,
) -> None:
    if not isinstance(metadata, dict) or not isinstance(metadata.get("index"), list):
        raise ValueError(f"FlashPack part {filename!r} has no valid footer index")
    actual_names: list[str] = []
    for record in metadata["index"]:
        if not isinstance(record, dict):
            raise ValueError(
                f"FlashPack part {filename!r} has an invalid footer record"
            )
        name = _require_nonempty_string(
            record.get("name"), f"{filename} footer tensor name"
        )
        shape = record.get("shape")
        offset = record.get("offset")
        length = record.get("length")
        if (
            not isinstance(shape, list)
            or any(
                not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
                for dim in shape
            )
            or not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or not isinstance(length, int)
            or isinstance(length, bool)
            or length <= 0
        ):
            raise ValueError(
                f"FlashPack part {filename!r} has invalid metadata for tensor {name!r}"
            )
        numel = 1
        for dim in shape:
            numel *= dim
        if numel != length:
            raise ValueError(
                f"FlashPack part {filename!r} tensor {name!r} shape has "
                f"{numel} elements but footer length is {length}"
            )
        actual_names.append(name)

    if len(actual_names) != len(set(actual_names)):
        raise ValueError(f"FlashPack part {filename!r} has duplicate tensor names")
    actual = set(actual_names)
    if actual != expected_names:
        missing = sorted(expected_names - actual)
        unexpected = sorted(actual - expected_names)
        raise ValueError(
            f"FlashPack part {filename!r} footer disagrees with manifest: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _import_flashpack():
    try:
        from flashpack.deserialization import (
            get_flashpack_file_metadata,
            iterate_from_flash_tensor,
            read_flashpack_file,
        )
    except ImportError as exc:
        raise ImportError(
            "FlashPack loading requires flashpack>=0.4.0. "
            "Install it with `pip install vllm[flashpack]`."
        ) from exc
    return (
        get_flashpack_file_metadata,
        read_flashpack_file,
        iterate_from_flash_tensor,
    )


def flashpack_weights_iterator(
    index: FlashPackIndex,
    parts: tuple[FlashPackPart, ...],
    resolve_part: Callable[[str], str],
    should_load_weight: Callable[[str], bool],
    *,
    device: torch.device,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield local weights while retaining at most one part's storage."""
    (
        get_flashpack_file_metadata,
        read_flashpack_file,
        iterate_from_flash_tensor,
    ) = _import_flashpack()
    names_by_part: dict[str, set[str]] = {}
    for name, filename in index.weight_map.items():
        names_by_part.setdefault(filename, set()).add(name)

    for part in parts:
        path = resolve_part(part.filename)

        # Footer validation happens before checksum streaming and payload
        # transfer. The checksum then authenticates both footer and payload.
        metadata = get_flashpack_file_metadata(path)
        _validate_footer_tensor_names(
            metadata, names_by_part[part.filename], part.filename
        )
        _validate_part_file(path, part)

        storage, metadata = read_flashpack_file(
            path,
            device=device,
            metadata=metadata,
        )
        try:
            tensor_iterator = iterate_from_flash_tensor(storage, metadata)
            for name, tensor in tensor_iterator:
                if should_load_weight(name):
                    yield name, tensor
                del tensor
            del tensor_iterator
        finally:
            # Views yielded for custom weight loaders are expected to be
            # consumed synchronously. Dropping the backing storage here keeps
            # peak transient memory bounded to one part.
            del storage


class FlashPackModelLoader(DefaultModelLoader):
    """Load manifest-sharded FlashPack parts through vLLM weight loaders."""

    def __init__(self, load_config: LoadConfig):
        # DefaultModelLoader's extra-config schema is intentionally specific
        # to HF/PT loading. Reuse its PP/EP initialization and weight audit,
        # but validate this loader's independent schema here.
        BaseModelLoader.__init__(self, load_config)
        self.local_expert_ids: set[int] | None = None
        self.local_layer_range: tuple[int, int] | None = None
        self.is_first_pipeline_rank = True
        self.is_last_pipeline_rank = True
        self.counter_before_loading_weights = 0.0
        self.counter_after_loading_weights = 0.0
        self.enable_weights_track = None

        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError("FlashPack model_loader_extra_config must be a dict")
        allowed_keys = {"include_mtp", "manifest_filename", "enable_weights_track"}
        unexpected_keys = set(extra_config) - allowed_keys
        if unexpected_keys:
            raise ValueError(
                f"Unexpected FlashPack model loader config keys: {unexpected_keys}"
            )
        include_mtp = extra_config.get("include_mtp", False)
        if not isinstance(include_mtp, bool):
            raise ValueError("FlashPack include_mtp must be a bool")
        self.include_mtp = include_mtp
        self.enable_weights_track = extra_config.get("enable_weights_track")
        if self.enable_weights_track not in (None, True, False):
            raise ValueError("FlashPack enable_weights_track must be a bool")

        manifest_filename = extra_config.get("manifest_filename", FLASHPACK_INDEX_NAME)
        self.manifest_filename = _require_nonempty_string(
            manifest_filename, "manifest_filename"
        )
        manifest_path = PurePosixPath(self.manifest_filename)
        if manifest_path.is_absolute() or ".." in manifest_path.parts:
            raise ValueError(
                f"Unsafe FlashPack manifest filename: {self.manifest_filename!r}"
            )

    def _resolve_file(self, model_config: ModelConfig, filename: str) -> str:
        if os.path.isdir(model_config.model):
            return str(Path(model_config.model) / filename)
        return hf_api().hf_hub_download(
            repo_id=model_config.model,
            filename=filename,
            cache_dir=self.load_config.download_dir,
            revision=model_config.revision,
        )

    def _load_index(self, model_config: ModelConfig) -> FlashPackIndex:
        path = self._resolve_file(model_config, self.manifest_filename)
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing FlashPack index {self.manifest_filename!r}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in FlashPack index {self.manifest_filename!r}"
            ) from exc
        return parse_flashpack_index(data)

    def download_model(self, model_config: ModelConfig) -> None:
        # Downloading the manifest only is deliberate. Each distributed rank
        # downloads its selected PP-local parts during load_weights.
        self._load_index(model_config)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # Import lazily: models.utils imports model_loader.reload, which first
        # initializes this package's __init__ and registers this loader.
        # Importing models.utils at module scope therefore forms a cycle before
        # is_pp_missing_parameter has been defined.
        from vllm.model_executor.models.utils import is_pp_missing_parameter

        index = self._load_index(model_config)
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()

        def should_load_weight(name: str) -> bool:
            return (
                not should_skip_pp_weight(
                    name,
                    self.local_layer_range,
                    is_first_pipeline_rank=self.is_first_pipeline_rank,
                    is_last_pipeline_rank=self.is_last_pipeline_rank,
                )
                and not should_skip_weight(name, self.local_expert_ids)
                and not is_pp_missing_parameter(name, model)
            )

        parts = select_flashpack_parts(
            index,
            should_load_weight,
            include_mtp=self.include_mtp,
            is_first_pipeline_stage=(
                self.local_layer_range is None or self.is_first_pipeline_rank
            ),
            is_last_pipeline_stage=(
                self.local_layer_range is None or self.is_last_pipeline_rank
            ),
        )
        if not parts:
            raise ValueError("FlashPack manifest selected no local model parts")

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device(self.load_config.device or "cpu")

        logger.info(
            "Selected %d/%d FlashPack parts for this pipeline rank (source %s@%s)",
            len(parts),
            len(index.parts),
            index.source_repo_id,
            index.source_revision,
        )
        yield from flashpack_weights_iterator(
            index,
            parts,
            lambda filename: self._resolve_file(model_config, filename),
            should_load_weight,
            device=device,
        )
