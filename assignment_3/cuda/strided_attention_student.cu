#include <torch/extension.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>

#include <c10/cuda/CUDAException.h>

// CUDA kernel for the forward pass of strided attention
//
// T: data type (e.g., float, double, half)
//
// Computes: output = Softmax((Q * K_strided^T) / sqrt(head_dim)) * V_strided
//
// Grid/Block Dimensions:
// - gridDim.x: batch_size * num_heads
// - gridDim.y: seq_len (for queries)
// - blockDim.x: Should be a power of 2, e.g., 128, 256. Represents threads per query token.
//
template <typename T>
__global__ void strided_attention_forward_kernel(
    const T* __restrict__ q_ptr,
    const T* __restrict__ k_ptr,
    const T* __restrict__ v_ptr,
    T* __restrict__ output_ptr,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int stride) {

    // =================================================================
    // TODO: IMPLEMENT YOUR STRIDED ATTENTION KERNEL HERE
    // =================================================================
    //
    // Hints:
    // 1. Calculate thread/block indices to figure out which output element
    //    this thread block is responsible for.
    //    - `blockIdx.x` can map to (batch, head).
    //    - `blockIdx.y` can map to the query token index (`i`).
    //    - `threadIdx.x` is the thread within the block.

    // 2. Load the query vector `q_i` for the current query token `i`.
    //    Since all threads in the block need it, you can load it into shared memory.

    // 3. Compute dot products `Q_i * K_j^T` for all `j` in the stride.
    //    - Parallelize the dot product calculation across threads in the block.
    //    - Each thread can compute a partial dot product and then you can use
    //      a parallel reduction (e.g., using shared memory and `__syncthreads()`)
    //      to get the final score.

    // 4. Compute Softmax.
    //    - Find the maximum score among the strided scores (for numerical stability). This is another parallel reduction.
    //    - Compute the exponential of each score (subtracting the max).
    //    - Compute the sum of the exponentials (another parallel reduction).
    //    - Normalize to get the final attention probabilities.

    // 5. Compute the weighted sum of value vectors `V_j`.
    //    - Use the attention probabilities to weight the `v_j` vectors.
    //    - This is another parallel dot-product-like operation. Each thread can
    //      compute a part of the final output vector component.

    // 6. Write the final output vector to `output_ptr`.

    // Example of getting indices (you'll need to expand on this):
    int batch_idx = blockIdx.x / num_heads;
    int head_idx = blockIdx.x % num_heads;
    int query_idx = blockIdx.y; // This is 'i' in the outer loop

    // Placeholder: just copy query to output to have a compilable stub
    long q_offset = (long)batch_idx * num_heads * seq_len * head_dim +
                    (long)head_idx * seq_len * head_dim +
                    (long)query_idx * head_dim;
    
    long output_offset = q_offset;
    
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        output_ptr[output_offset + i] = q_ptr[q_offset + i];
    }
}


// C++ function that dispatches the CUDA kernel
torch::Tensor strided_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int stride) {

    // Validate inputs
    TORCH_CHECK(q.is_cuda(), "Input tensor Q must be on a CUDA device");
    TORCH_CHECK(q.is_contiguous(), "Input tensor Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input tensor K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input tensor V must be contiguous");

    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_dim = q.size(3);

    // Create an output tensor of the same shape as Q
    auto output = torch::empty_like(q);

    // Define grid and block dimensions
    // Grid: One block per (batch, head, query_token)
    dim3 gridDim(batch_size * num_heads, seq_len);
    // Block: Threads to parallelize the work for a single query token
    dim3 blockDim(256); // A common choice, can be tuned

    // Dispatch the kernel based on the data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "strided_attention_forward", ([&] {
        strided_attention_forward_kernel<scalar_t><<<gridDim, blockDim>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            stride
        );
    }));

    // Check for any CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}


// Bind the C++ function to a Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("strided_attention_forward", &strided_attention_forward_cuda, "Strided Attention Forward (CUDA)");
}
