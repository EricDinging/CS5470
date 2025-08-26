# Cornell University - Homework 2: Scheduling Prompts

## Course Information
- **Course:** [COURSE NUMBER]
- **Assignment:** Homework 2: Scheduling prompts
- **Due Date:** [Yesterday]

## Overview

This homework focuses on understanding the GPU memory capacity requirements to serve Large Language Models (LLMs) using the state-of-the-art serving engine **vLLM**. You will:

- Set up a 34B model on 1 A100 GPU
- Measure memory occupation of weights
- Determine remaining memory for KV cache
- Benchmark how request bursts affect LLM responsiveness
- Implement scheduler improvements to enhance system responsiveness

## Prerequisites

### Hardware Access
- **Platform:** Perlmutter HPC
- **Resources:** Server with 4 A100 GPUs (80 GB memory each)
- **Setup:** Follow the email instructions to reserve servers

### Software Setup

#### Step 1: Setup vLLM
Complete the vLLM setup as outlined in Homework 1. This is a prerequisite for the current assignment.

#### Step 2: Homework Files
1. Copy the homework zip file into the vLLM source root directory
2. Uncompress the file
3. Verify creation of the `assignment_1` directory

## Homework Tasks

This assignment is designed to help you understand:
- How serving engines schedule prompts on GPU
- Methods to improve serving responsiveness

### Task 1: Understanding High Time-To-First-Token (TTFT)

The homework folder includes:
- `server.sh`: Script that starts a vLLM instance running **codeLlama-34B** on GPU-0
- Exposes an OpenAI-compatible API endpoint

## Getting Started

1. Ensure you have completed Homework 1 (vLLM setup)
2. Reserve a Perlmutter HPC server following the provided instructions
3. Copy and extract the homework files to the vLLM source root
4. Navigate to the `assignment_1` directory
5. Run the provided `server.sh` script to start the vLLM instance

## Expected Outcomes

By the end of this homework, you should understand:
- GPU memory allocation for LLM serving
- Impact of request patterns on system performance
- Scheduler optimization techniques for improved responsiveness

---

*Note: Replace [COURSE NUMBER] and [Yesterday] with actual course information and due date.*