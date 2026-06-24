# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere (sm_86) DeepSeek V4 backend: all-triton kernels (indexer MQA-logits
incl. int8 IMMA, sparse-MLA decode/prefill, fp8/int8 cache) selected when the
device capability is sm_8x. Parallel to nvidia/ (Hopper/Blackwell)."""
