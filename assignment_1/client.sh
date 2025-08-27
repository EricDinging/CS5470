#!/bin/bash

python3 benchmark.py --backend vllm \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --request-rate 2 \
    --num-prompts 25 \
    --dataset-name dummy \
    --long-prompts 0 \
    --long-prompt-len 32000