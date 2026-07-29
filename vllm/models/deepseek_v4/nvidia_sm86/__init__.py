# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility alias for the module now named ``nvidia_imma``.

Published checkpoints name kernels by fully-qualified symbol in their
``"vllm"`` config block, and several of those FQNs are rooted at
``vllm.models.deepseek_v4.nvidia_sm86.triton_kernels``. A checkpoint is a
released artifact that cannot be edited retroactively, so the old path has to
keep resolving to the same objects.

The package was renamed because ``sm86`` names the architecture it was
written on rather than what it requires: IMMA plus ``cp.async``, i.e. sm_80
and up, which includes sm_12x. Selecting it by arch equality made an
``int8_ds_mla`` checkpoint unservable on GB10.
"""

import sys

from vllm.models.deepseek_v4 import nvidia_imma
from vllm.models.deepseek_v4.nvidia_imma import *  # noqa: F401,F403
from vllm.models.deepseek_v4.nvidia_imma import attention, triton_kernels

# Bind the submodules under the legacy package name as well, so that an
# importlib resolution of "vllm.models.deepseek_v4.nvidia_sm86.triton_kernels"
# -- how kernel_config resolves a checkpoint's FQN -- finds the same module
# object rather than importing a second copy.
sys.modules[__name__ + ".attention"] = attention
sys.modules[__name__ + ".triton_kernels"] = triton_kernels

__all__ = getattr(nvidia_imma, "__all__", [])
