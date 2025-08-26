#include <torch/extension.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>
#include <cmath>

#include <c10/cuda/CUDAException.h>

template <typename T>
__device__ __forceinline__ T my_exp(T x) { return exp(x); }
template <>
__device__ __forceinline__ float my_exp<float>(float x) { return expf(x); }
template <>
__device__ __forceinline__ __half my_exp<__half>(__half x) {
  float f = __half2float(x);
  return __float2half(expf(f));
}

// Device function to perform a parallel reduction within a block using shared memory.
// This function operates directly on a provided shared memory buffer
// to perform an in-place reduction. This avoids all declaration conflicts.
template <typename T>
__device__ T block_reduce_sum(T* shared_data, T val) {
    shared_data[threadIdx.x] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    return shared_data[0];
}

// Device function to find the maximum value in a block using a parallel reduction.
// This function also operates in-place on a provided buffer.
template <typename T>
__device__ T block_reduce_max(T* shared_data, T val) {
    shared_data[threadIdx.x] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] = max(shared_data[threadIdx.x], shared_data[threadIdx.x + s]);
        }
        __syncthreads();
    }
    return shared_data[0];
}

// CUDA kernel for the forward pass of strided attention
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

    extern __shared__ unsigned char _shmem_bytes[];
    T* smem = reinterpret_cast<T*>(_shmem_bytes);

    T* shared_q      = smem;
    T* shared_scores = smem + head_dim;

    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int query_idx = blockIdx.y;

    const float scale = T(1) / sqrt(T(head_dim));
    const int num_strided_keys = (seq_len + stride - 1) / stride;

    const long bh_offset = (long)(batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const T* q_base_ptr = q_ptr + bh_offset;
    const T* k_base_ptr = k_ptr + bh_offset;
    const T* v_base_ptr = v_ptr + bh_offset;
    T* output_base_ptr = output_ptr + bh_offset;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        shared_q[i] = q_base_ptr[query_idx * head_dim + i];
    }
    __syncthreads();

    // Each thread calculates a score if it's within the strided key count
    T score = T(0);
    if (threadIdx.x < num_strided_keys) {
        const int key_idx = threadIdx.x * stride;
        for(int i = 0; i < head_dim; ++i) {
            score += shared_q[i] * k_base_ptr[key_idx * head_dim + i];
        }
        score *= scale;
        shared_scores[threadIdx.x] = score;
    }
    __syncthreads();

    // Prepare a value for EVERY thread in the block, initializing to a
    // safe default if the thread is outside the range of valid scores.
    // This makes the subsequent call to the self-contained reduction safe.
    T max_score = block_reduce_max(shared_scores, score);
    __syncthreads();

    T exp_sum_val = T(0);
    if (threadIdx.x < num_strided_keys) {
        exp_sum_val = my_exp(score-max_score);
        shared_scores[threadIdx.x] = exp_sum_val;
    }
    // The reduction buffer for sum is the same as the score buffer
    T exp_sum = block_reduce_sum(shared_scores, exp_sum_val);
    __syncthreads();

    T inv_exp_sum = T(1) / (exp_sum + T(1e-6));
    if (threadIdx.x < num_strided_keys) {
        shared_scores[threadIdx.x] = exp_sum_val*inv_exp_sum;
    }
    __syncthreads();

    if (threadIdx.x < head_dim) {
        T sum = T(0);
        // for each “strided” key row, grab its base pointer then index [threadIdx.x]
        for (int i = 0; i < num_strided_keys; ++i) {
            const T* v_row = v_base_ptr + (i * stride) * head_dim;
            sum += shared_scores[i] * v_row[threadIdx.x];
        }
        // write back:
        output_base_ptr[query_idx * head_dim + threadIdx.x] = sum;
    }
}

// C++ function that dispatches the CUDA kernel
torch::Tensor strided_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int stride) {

    TORCH_CHECK(q.is_cuda(), "Input tensor Q must be on a CUDA device");
    TORCH_CHECK(q.is_contiguous(), "Input tensor Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input tensor K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input tensor V must be contiguous");

    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_dim = q.size(3);

    auto output = torch::empty_like(q);

    // FIX: Define the lambda function `num_strided_keys` *before* it is used.
    const int num_strided_keys = (seq_len + stride - 1) / stride;

    dim3 blockDim(256);
    dim3 gridDim(batch_size * num_heads, seq_len);

    // Shared memory for the main kernel is just for q_vec and scores
    //const int elems = head_dim + num_strided_keys(seq_len, stride);
    //const int shared_mem_size = elems * q.element_size();
    const int block_size = blockDim.x;
    // allocate enough T’s for both Q and the *entire* blockDim.x worth of scores
    const int shared_elems = head_dim + block_size;
    const int shared_mem_size = shared_elems * q.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "strided_attention_forward", ([&] {
        // The shared memory size is dynamically calculated based on the input tensor type.
        strided_attention_forward_kernel<scalar_t><<<gridDim, blockDim, shared_mem_size>>>(
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

    C10_CUDA_CHECK(cudaGetLastError());

    return output;
}


// Bind the C++ function to a Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("strided_attention_forward", &strided_attention_forward_cuda, "Strided Attention Forward (CUDA)");
}
