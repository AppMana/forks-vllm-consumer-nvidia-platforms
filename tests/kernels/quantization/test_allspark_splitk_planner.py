# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import os
from pathlib import Path


class _BlockTileSplitkParams(ctypes.Structure):
    _fields_ = [
        ("m_tile", ctypes.c_int),
        ("n_tile", ctypes.c_int),
        ("split_k", ctypes.c_int),
        ("enable_fuse", ctypes.c_bool),
    ]


def _planner_library() -> Path:
    override = os.environ.get("VLLM_ALLSPARK_PLANNER_LIBRARY")
    if override is not None:
        return Path(override)

    root = Path(__file__).parents[3]
    libraries = list((root / "vllm").glob("_C_stable_libtorch*.so"))
    assert len(libraries) == 1, libraries
    return libraries[0]


def _plan(m: int, n: int, k: int, sm_count: int):
    library = ctypes.CDLL(str(_planner_library()))
    planner = library[
        "_ZN8allspark54allspark_qgemm_w8a16_perc_n32k16_ampere_"
        "workspace_sizeEiiiiRNS_21BlockTileSplitkParamsE"
    ]
    planner.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_BlockTileSplitkParams),
    ]
    planner.restype = ctypes.c_size_t

    params = _BlockTileSplitkParams()
    workspace_size = planner(m, n, k, sm_count, ctypes.byref(params))
    return params, workspace_size


def test_splitk_planner_uses_progress_safe_reduction():
    m, n, k, sm_count = 1024, 384, 4096, 84

    assert ctypes.sizeof(_BlockTileSplitkParams) == 16
    params, workspace_size = _plan(m, n, k, sm_count)
    grid_z = (k + params.split_k - 1) // params.split_k

    assert (params.m_tile, params.n_tile, params.split_k) == (64, 128, 1376)
    assert grid_z == 3
    assert not params.enable_fuse
    assert workspace_size == grid_z * m * n * 2
