#!/bin/bash
# L3 two-node TP=2 real-weights serve (SSH prototype of the LWS deployment).
# Usage: serve_l3.sh <node-rank 0|1>
set -e
RANK=$1
# GB10 page-cache reclaim (the LWS initContainer pattern): mmapped safetensors
# from prior loads/copies otherwise eat into the conservative startup check.
sudo sysctl -qw vm.drop_caches=3
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export VLLM_HOST_IP=10.255.0.$((RANK+1))
export VLLM_LOGGING_LEVEL=INFO
ARGS=(
  "$HOME/workspace/deepseek-v4-nvfp4-fp8"
  --tensor-parallel-size 2
  --distributed-executor-backend mp
  --nnodes 2 --node-rank "$RANK"
  --master-addr 10.255.0.1 --master-port 29501
  --tokenizer-mode deepseek_v4
  --trust-remote-code
  --max-num-seqs 4
  --max-model-len 8192
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "greedy"}'
  --kv-cache-memory-bytes 12884901888
  --port 8000
  --gpu-memory-utilization 0.80
)
[ "$RANK" = "1" ] && ARGS+=(--headless)
exec "$HOME/workspace/venv/bin/vllm" serve "${ARGS[@]}"
