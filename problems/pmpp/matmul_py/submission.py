from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import torch

mm_cuda_source = """
template <typename scalar_t>
#define TILE_DIM 32
__global__ void mm_kernel(const scalar_t* __restrict__ a, 
                           const scalar_t* __restrict__ b, 
                           scalar_t* __restrict__ P,
                           int m,
                           int k,
                           int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ scalar_t a_s[TILE_DIM][TILE_DIM];
    __shared__ scalar_t b_s[TILE_DIM][TILE_DIM];

    float sum = 0.0f;
    for (int i = 0; i < (k + TILE_DIM - 1) / TILE_DIM; ++i) {
        if (row < m && i * TILE_DIM + threadIdx.x < k) {
            a_s[threadIdx.y][threadIdx.x] = a[row * k + i * TILE_DIM + threadIdx.x];
        } else {
            a_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && i * TILE_DIM + threadIdx.y < k) {
            b_s[threadIdx.y][threadIdx.x] = b[(i * TILE_DIM + threadIdx.y) * n + col];
        } else {
            b_s[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < TILE_DIM; ++j) {
            sum += a_s[threadIdx.y][j] * b_s[j][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        P[row * n + col] = sum; 
    }
}

torch::Tensor mm_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    
    int m = torch::size(a, 0);
    int k = torch::size(a, 1);
    int n = torch::size(b, 1);
    TORCH_CHECK(k == torch::size(b, 0), "Inner dimensions of a and b must match");

    auto P = torch::zeros({m, n}, a.options()); 

    dim3 threads(TILE_DIM, TILE_DIM, 1);
    dim3 blocks((n + TILE_DIM - 1) / TILE_DIM, 
                (m + TILE_DIM - 1) / TILE_DIM, 
                 1); 
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "mm_kernel", ([&] {
        mm_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            P.data_ptr<scalar_t>(),
            m,
            k,
            n
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return P;
}
"""

mm_cpp_source = """
#include <torch/extension.h>

torch::Tensor mm_cuda(torch::Tensor a , torch::Tensor b);
"""

mm_module = load_inline(
    name='mm_cuda',
    cpp_sources=mm_cpp_source,
    cuda_sources=mm_cuda_source,
    functions=['mm_cuda'],
    verbose=True,
)

def mm(a, b):
    if not a.is_cuda or not b.is_cuda:
        raise RuntimeError("Both tensors must be on GPU")
    return mm_module.mm_cuda(a, b)

def custom_kernel(data: input_t) -> output_t:
    a, b = data
    return mm(a, b)
