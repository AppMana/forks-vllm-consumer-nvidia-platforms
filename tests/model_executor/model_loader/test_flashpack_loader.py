# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import hashlib
import json
import subprocess
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch
from flashpack import pack_to_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.flashpack_loader import (
    FLASHPACK_INDEX_FORMAT,
    FlashPackModelLoader,
    FlashPackPart,
    _validate_footer_tensor_names,
    flashpack_weights_iterator,
    parse_flashpack_index,
    select_flashpack_parts,
)
from vllm.model_executor.model_loader.pp_weight_filter import (
    should_skip_pp_weight,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_part(path, tensors):
    pack_to_file(tensors, str(path), target_dtype=None)
    return {
        "sha256": _sha256(path),
        "size": path.stat().st_size,
        "kind": "model",
    }


def _manifest(parts, weight_map):
    return {
        "format": FLASHPACK_INDEX_FORMAT,
        "source": {
            "repo_id": "deepseek-ai/DeepSeek-V4-Flash",
            "revision": "0123456789abcdef",
        },
        "parts": parts,
        "weight_map": weight_map,
    }


def test_manifest_validation_is_strict():
    good = _manifest(
        {
            "model-00001.flashpack": {
                "sha256": "a" * 64,
                "size": 123,
                "kind": "model",
            }
        },
        {"layers.0.weight": "model-00001.flashpack"},
    )
    index = parse_flashpack_index(good)
    assert index.source_revision == "0123456789abcdef"
    assert index.parts[0].size == 123

    missing_revision = json.loads(json.dumps(good))
    del missing_revision["source"]["revision"]
    with pytest.raises(ValueError, match="source.revision"):
        parse_flashpack_index(missing_revision)

    undeclared_part = json.loads(json.dumps(good))
    undeclared_part["weight_map"]["layers.0.weight"] = "missing.flashpack"
    with pytest.raises(ValueError, match="undeclared"):
        parse_flashpack_index(undeclared_part)

    unsafe_filename = json.loads(json.dumps(good))
    unsafe_filename["parts"]["../model.flashpack"] = unsafe_filename["parts"].pop(
        "model-00001.flashpack"
    )
    unsafe_filename["weight_map"]["layers.0.weight"] = "../model.flashpack"
    with pytest.raises(ValueError, match="Unsafe"):
        parse_flashpack_index(unsafe_filename)


def test_flashpack_load_format_is_registered():
    assert isinstance(
        get_model_loader(LoadConfig(load_format="flashpack")),
        FlashPackModelLoader,
    )


def test_flashpack_registration_does_not_cycle_model_utils_import():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm.model_executor.models.utils import is_pp_missing_parameter",
        ],
        check=True,
    )


def test_pp2_selects_parts_before_opening_payloads():
    index = parse_flashpack_index(
        _manifest(
            {
                "embed.flashpack": {
                    "sha256": "a" * 64,
                    "size": 1,
                    "kind": "model",
                    "pipeline_stage": "first",
                },
                "layer-0.flashpack": {
                    "sha256": "b" * 64,
                    "size": 1,
                    "kind": "model",
                },
                "layer-1.flashpack": {
                    "sha256": "c" * 64,
                    "size": 1,
                    "kind": "model",
                },
                "mtp.flashpack": {
                    "sha256": "d" * 64,
                    "size": 1,
                    "kind": "mtp",
                },
                "head.flashpack": {
                    "sha256": "e" * 64,
                    "size": 1,
                    "kind": "model",
                    "pipeline_stage": "last",
                },
            },
            {
                "embed_tokens.weight": "embed.flashpack",
                "layers.0.weight": "layer-0.flashpack",
                "layers.1.weight": "layer-1.flashpack",
                "layers.2.weight": "mtp.flashpack",
                "head.weight": "head.flashpack",
            },
        )
    )

    rank0 = select_flashpack_parts(
        index,
        lambda name: not should_skip_pp_weight(name, (0, 1)),
        include_mtp=False,
        is_first_pipeline_stage=True,
        is_last_pipeline_stage=False,
    )
    rank1 = select_flashpack_parts(
        index,
        lambda name: not should_skip_pp_weight(name, (1, 2)),
        include_mtp=False,
        is_first_pipeline_stage=False,
        is_last_pipeline_stage=True,
    )
    assert [part.filename for part in rank0] == [
        "embed.flashpack",
        "layer-0.flashpack",
    ]
    assert [part.filename for part in rank1] == [
        "layer-1.flashpack",
        "head.flashpack",
    ]
    assert all(part.kind != "mtp" for part in (*rank0, *rank1))


def test_custom_model_load_weights_receives_original_names_and_dtypes(
    tmp_path, monkeypatch
):
    part_path = tmp_path / "model-00001.flashpack"
    tensors = {
        "layers.0.int4_packed": torch.tensor([[0, 15]], dtype=torch.uint8),
        "layers.0.int8_weight": torch.tensor([[-128, 127]], dtype=torch.int8),
    }
    part = _write_part(part_path, tensors)
    (tmp_path / "model.flashpack.index.json").write_text(
        json.dumps(
            _manifest(
                {"model-00001.flashpack": part},
                {name: "model-00001.flashpack" for name in tensors},
            )
        )
    )

    class CustomWeightModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.called = 0
            self.loaded = {}

        def load_weights(self, weights):
            self.called += 1
            for name, tensor in weights:
                self.loaded[name] = tensor.clone()
            return set(self.loaded)

    loader = FlashPackModelLoader(LoadConfig(load_format="flashpack"))
    monkeypatch.setattr(loader, "_init_ep_weight_filter", lambda model_config: None)
    monkeypatch.setattr(loader, "_init_pp_weight_filter", lambda model_config: None)
    monkeypatch.setattr(loader, "track_weights_loading", lambda *args, **kwargs: None)
    model = CustomWeightModel()
    model_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        quantization=None,
    )

    loader.load_weights(model, model_config)

    assert model.called == 1
    assert set(model.loaded) == set(tensors)
    assert model.loaded["layers.0.int4_packed"].dtype == torch.uint8
    assert model.loaded["layers.0.int8_weight"].dtype == torch.int8
    torch.testing.assert_close(
        model.loaded["layers.0.int4_packed"], tensors["layers.0.int4_packed"]
    )
    torch.testing.assert_close(
        model.loaded["layers.0.int8_weight"], tensors["layers.0.int8_weight"]
    )


def test_disabled_mtp_part_is_never_resolved(tmp_path):
    model_path = tmp_path / "model.flashpack"
    mtp_path = tmp_path / "mtp.flashpack"
    model_part = _write_part(
        model_path, {"layers.0.weight": torch.ones(1, dtype=torch.int8)}
    )
    mtp_part = _write_part(
        mtp_path, {"mtp.layers.0.weight": torch.ones(1, dtype=torch.int8)}
    )
    mtp_part["kind"] = "mtp"
    index = parse_flashpack_index(
        _manifest(
            {
                model_path.name: model_part,
                mtp_path.name: mtp_part,
            },
            {
                "layers.0.weight": model_path.name,
                "mtp.layers.0.weight": mtp_path.name,
            },
        )
    )
    parts = select_flashpack_parts(index, lambda name: True, include_mtp=False)
    resolved = []

    def resolve(filename):
        resolved.append(filename)
        return str(tmp_path / filename)

    assert [
        name
        for name, _ in flashpack_weights_iterator(
            index,
            parts,
            resolve,
            lambda name: True,
            device=torch.device("cpu"),
        )
    ] == ["layers.0.weight"]
    assert resolved == [model_path.name]


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_missing_or_corrupt_part_fails_closed_before_payload(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / "part.flashpack"
    path.write_bytes(b"valid-looking-part")
    part = FlashPackPart(
        filename=path.name,
        sha256=hashlib.sha256(b"valid-looking-part").hexdigest(),
        size=path.stat().st_size,
        kind="model",
        pipeline_stage="any",
    )
    index = parse_flashpack_index(
        _manifest(
            {
                path.name: {
                    "sha256": part.sha256,
                    "size": part.size,
                    "kind": "model",
                }
            },
            {"layers.0.weight": path.name},
        )
    )
    metadata = {
        "index": [
            {
                "name": "layers.0.weight",
                "shape": [1],
                "offset": 0,
                "length": 1,
            }
        ]
    }
    payload_reads = 0

    def read_payload(*args, **kwargs):
        nonlocal payload_reads
        payload_reads += 1
        raise AssertionError("payload must not be read")

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.flashpack_loader._import_flashpack",
        lambda: (lambda path: metadata, read_payload, lambda storage, meta: ()),
    )
    if failure == "missing":
        path.unlink()
    else:
        path.write_bytes(b"x" * part.size)

    iterator = flashpack_weights_iterator(
        index,
        index.parts,
        lambda filename: str(tmp_path / filename),
        lambda name: True,
        device=torch.device("cpu"),
    )
    expected = FileNotFoundError if failure == "missing" else ValueError
    with pytest.raises(expected):
        next(iterator)
    assert payload_reads == 0


def test_footer_manifest_disagreement_fails_closed():
    with pytest.raises(ValueError, match="disagrees"):
        _validate_footer_tensor_names(
            {
                "index": [
                    {
                        "name": "unexpected",
                        "shape": [1],
                        "offset": 0,
                        "length": 1,
                    }
                ]
            },
            {"expected"},
            "part.flashpack",
        )


def test_parts_are_released_sequentially(tmp_path, monkeypatch):
    parts = {}
    weight_map = {}
    metadata_by_path = {}
    for part_number in range(2):
        path = tmp_path / f"part-{part_number}.flashpack"
        path.write_bytes(f"part-{part_number}".encode())
        name = f"layers.{part_number}.weight"
        parts[path.name] = {
            "sha256": _sha256(path),
            "size": path.stat().st_size,
            "kind": "model",
        }
        weight_map[name] = path.name
        metadata_by_path[str(path)] = {
            "index": [
                {
                    "name": name,
                    "shape": [1],
                    "offset": 0,
                    "length": 1,
                }
            ]
        }

    index = parse_flashpack_index(_manifest(parts, weight_map))
    previous_storage = None
    previous_tensor = None

    class Storage:
        def __init__(self, value):
            self.tensor = torch.tensor([value])

    def get_metadata(path):
        return metadata_by_path[path]

    def read_part(path, **kwargs):
        nonlocal previous_storage, previous_tensor
        gc.collect()
        if previous_storage is not None:
            assert previous_storage() is None
            assert previous_tensor() is None
        storage = Storage(len(path))
        previous_storage = weakref.ref(storage)
        previous_tensor = weakref.ref(storage.tensor)
        return storage, kwargs["metadata"]

    def iterate(storage, metadata):
        yield metadata["index"][0]["name"], storage.tensor

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.flashpack_loader._import_flashpack",
        lambda: (get_metadata, read_part, iterate),
    )
    iterator = flashpack_weights_iterator(
        index,
        index.parts,
        lambda filename: str(tmp_path / filename),
        lambda name: True,
        device=torch.device("cpu"),
    )

    first_name, first_tensor = next(iterator)
    assert first_name == "layers.0.weight"
    del first_tensor
    second_name, second_tensor = next(iterator)
    assert second_name == "layers.1.weight"
    del second_tensor
    with pytest.raises(StopIteration):
        next(iterator)
