#!/bin/bash

CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server --model meta-llama/CodeLlama-34b-hf --swap-space 16 --disable-log-requests --enforce-eager --max-num-seqs 512 --disable-sliding-window  --load-format dummy --preemption-mode swap

# 70B model is here - https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct