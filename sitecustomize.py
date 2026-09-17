# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Testbed-only: applies the vllm_flash_attn stub to EVERY python started with
# this directory on PYTHONPATH, including vllm's model-inspection and worker
# subprocesses, which re-import vllm and would otherwise die on the fa2/fa3
# torch-ABI mismatch. Gated on the same env var as the harness stub.
import os

if os.environ.get("DSV4_STUB_VLLM_FA") == "1":
    import sys
    import types

    def _unavailable(*args, **kwargs):
        raise RuntimeError("vllm_flash_attn stubbed out by sitecustomize")

    _mod = types.ModuleType("vllm.vllm_flash_attn")
    _mod.FA2_AVAILABLE = False
    _mod.FA3_AVAILABLE = False
    _mod.flash_attn_varlen_func = _unavailable
    _mod.get_scheduler_metadata = _unavailable
    _mod.compile_flash_attn_varlen_func_from_specs = _unavailable
    _mod.fa_version_unsupported_reason = lambda v: "stubbed"
    _mod.is_fa_version_supported = lambda v: False
    sys.modules["vllm.vllm_flash_attn"] = _mod
