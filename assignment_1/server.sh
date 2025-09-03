#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --swap-space 16 \
    --disable-log-requests \
    --enforce-eager \
    --enable-chunked-prefill \
    --max-num-batched-tokens 512 \
    --max-num-seqs 512 \
    --disable-sliding-window
