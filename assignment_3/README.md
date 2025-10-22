# CS5470 - HW3: Custom CUDA Attention Kernel for a Transformer
**Due: 10/30/2025 11:59pm**

Welcome to the custom attention assignment! In this project, you will implement a custom attention mechanism, **Strided Attention**, and accelerate it by writing your own CUDA kernel.

## Project Goal

The goal is to replace a standard, naive PyTorch attention implementation with a high-performance version written in CUDA. This will involve:
1.  Understanding the mathematics of Strided Attention.
2.  Designing and implementing a CUDA kernel to perform this operation efficiently.
3.  Using PyTorch's C++ extension functionality to compile and link your CUDA code.
4.  Integrating your custom kernel into a simplified Transformer model.
5.  Testing for correctness and benchmarking performance.

## File Structure

Here's an overview of the skeleton code provided, inside `src/`:

* `model.py`: Contains a simplified Transformer-like model. You won't need to edit this file, but you should understand how it uses the attention layer.
* `attention_student.py`: This is where the main logic for the attention mechanism resides.
    * `naive_strided_attention`: A reference implementation of strided attention in pure PyTorch. Use this to verify the correctness of your CUDA kernel.
    * `CustomStridedAttention`: A class where you will call your custom CUDA kernel. You need to complete the `forward` method here.
* `test.py`: A testing suite.
    * It verifies that your CUDA implementation produces the same output as the naive PyTorch version.
    * It includes a simple benchmark to compare the performance.
* `cuda/`: This directory contains all CUDA-related code.
    * `strided_attention_student.cu`: **This is the main file you will edit.** It contains the skeleton for your CUDA kernel. You need to fill in the implementation logic.
    * `binding.py`: This file handles the bridge between PyTorch and your CUDA code. It uses `torch.utils.cpp_extension` to load, compile, and create a Python binding for your C++/CUDA functions. You shouldn't need to edit this.

## Your Task: Strided Attention

Standard attention allows every token in a sequence to attend to every other token. Strided attention is a sparse attention pattern where each query token attends only to key tokens at a fixed `stride`.

For example, with a stride of 2, query `q_i` attends to keys `k_0, k_2, k_4, ...`.

### Steps to Complete

1.  **Understand the Code**: Read through all the provided files to understand how they fit together. Pay close attention to the `naive_strided_attention` function in `attention_student.py` to understand the logic you need to replicate.
2.  **Implement the CUDA Kernel**: Open `cuda/strided_attention_student.cu` and complete the `strided_attention_forward_kernel`. This is the core of the assignment. Think carefully about:
    * **Thread and Block Indexing**: How do you map threads to the output matrix elements?
    * **Memory Access**: How can you read from global memory (Q, K, V tensors) efficiently? Can you use shared memory to reduce global memory reads? (Using shared memory is an advanced goal, but highly encouraged for performance).
    * **Parallel Reduction**: The dot product and the final weighted sum of values are reduction operations. How do you perform these in parallel?
3.  **Integrate the Kernel**: In `attention_student.py`, complete the `forward` method of the `CustomStridedAttention` class to call your compiled CUDA kernel.
4.  **Test for Correctness**: Run `python test.py --mode=correctness`. This will run a small test case and assert that the output of your kernel matches the naive PyTorch implementation. Debug any discrepancies.
5.  **Benchmark Performance**: Once your implementation is correct, run `python test.py --mode=benchmark`. This will run a larger test case and report the execution time of both the naive and your custom implementation.

## Preparation

```
conda create --name sysml3 python=3.10.12
conda activate sysml3
pip install torch numpy
```

## How to Run


When you first run the `test.py` script, PyTorch's C++ extension will automatically compile your CUDA code. This may take a minute. Subsequent runs will be much faster as it will use the cached compiled library.

```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account m4999
# Test for correctness
python test.py --mode=correctness

# Benchmark performance
python test.py --mode=benchmark
```

You should ideally see a speed-up around 2x or better.

## Deliverables
1. Correctness (60%)
2. Performance (40%)
    - Target speedup 1.8x on hidden test cases
    - 10 points will be deducted for every 0.1 below 1.8 (i.e. deduction = (1.8 - your speedup) / 0.1 * 10)

## Submission
- Please submit your code on Perlmutter. You don't need to submit a report or execution results to Canvas.
    - `cp -r <your_hw3_folder> /global/cfs/cdirs/m4999/hw3/<netid>`
    - Your hw3 folder should contain:
        - `cuda/binding.py`
        - `cuda/strided_attention_student.cu`
        - `attention_student.py`
        - `model.py`
- We will run your code using the provided test case and several hidden test cases. The hidden test cases will use different model dimensions. You can assume that the provided block dimension, 256, is greater or equal to `head_dim` and `ceil(seq_len/stride)`
