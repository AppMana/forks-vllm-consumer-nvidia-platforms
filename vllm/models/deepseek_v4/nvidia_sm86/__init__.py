# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere (sm_86) DeepSeek V4 backend.

The sm86 attention class selects sparse MLA decode/prefill callables from the
DeepSeek V4 HF config. The indexer and linear/MoE paths are selected by their
own vLLM quantization and platform dispatch code.
"""
