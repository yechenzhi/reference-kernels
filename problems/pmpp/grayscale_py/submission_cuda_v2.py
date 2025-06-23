# H100 1587 µs with (1024, 1, 1) 
# H100 1458 µs with coarse factor 2
# H100 1645 µs with coarse factor 4
import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t

gray_cuda_source = """
template <typename scalar_t>
#define COARSE_FACTOR 4
__global__ void gray_kernel(const scalar_t* __restrict__ data,
                           const scalar_t* __restrict__ weights, 
                           scalar_t* __restrict__ C, 
                           int N) {
    int row_st = blockIdx.y * blockDim.y + threadIdx.y;
    int col_st = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;

    for(int i = 0; i < COARSE_FACTOR; ++i) {
        int row = row_st;
        int col = col_st + i;

        if (row < N && col < N) {
            scalar_t sum = 0.0f;
            for (int k = 0; k < 3; ++k) {
                sum += data[row * N * 3 + col * 3 + k] * weights[k];
            }
            C[row * N + col] = sum;   
        }
    }
    
}

torch::Tensor gray_cuda(torch::Tensor data, torch::Tensor weights) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");
    TORCH_CHECK(weights.device().is_cuda(), "Tensor weights must be a CUDA tensor");
    
    int N = torch::size(data, 0);  
    auto C = torch::empty({N, N}, data.options()); 

    dim3 threads(1024, 1, 1);
    dim3 blocks((N + threads.x * COARSE_FACTOR - 1) / (threads.x * COARSE_FACTOR), (N + threads.y - 1) / threads.y, 1);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "gray_kernel", ([&] {
        gray_kernel<scalar_t><<<blocks, threads>>>(
            data.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            N
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

gray_cpp_source = """
#include <torch/extension.h>

torch::Tensor gray_cuda(torch::Tensor data, torch::Tensor weights);
"""

gray_module = load_inline(
    name='gray_cuda',
    cpp_sources=gray_cpp_source,
    cuda_sources=gray_cuda_source,
    functions=['gray_cuda'],
    verbose=True,
)

def grayscale(data, weights):
    if not data.is_cuda or not weights.is_cuda:
        raise RuntimeError("Both tensors must be on GPU")
    return gray_module.gray_cuda(data, weights)

def custom_kernel(data: input_t) -> output_t:

    weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                         device=data.device, 
                         dtype=data.dtype)

    assert data.is_cuda and weights.is_cuda, "Input tensors must be on GPU"
    
    # Simply reuse the existing add function we already defined
    # This avoids the compilation issues with the inline kernel
    return grayscale(data, weights)
