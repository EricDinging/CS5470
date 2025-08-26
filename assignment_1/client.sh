#!/bin/bash

python3 benchmark.py --backend openai --model meta-llama/Llama-3.1-8B-Instruct --request-rate 2 --num-prompts 25 --dataset-name dummy --long-prompts 0 --long-prompt-len 32000