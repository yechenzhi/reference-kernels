import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#define BLOCK_DIM 1024
#define COARSE_FACTOR1 2
#define COARSE_FACTOR2 16
#define COARSE_THREASHOLD 512
template <typename scalar_t>
__device__ int coRank(const scalar_t* A, const scalar_t* B, int m, int n, int k) {
    int iLow = (k > n) ? (k - n) : 0;
    int iHigh = (k < m) ? k : m;

    while(iLow <= iHigh){
        int i = iLow + (iHigh - iLow) / 2;
        int j = k - i;
        if(i > 0 && j < n && A[i - 1] > B[j]) {
            iHigh = i - 1;
        } else if(j > 0 && i < m && B[j - 1] >= A[i]) {
            iLow = i + 1;
        } else {
            return i;
        }
    }
    return iLow;
}

template <typename scalar_t>
__device__ void mergeSequential(const scalar_t* A, const scalar_t* B, scalar_t* C, int m, int n) {
    int i = 0;
    int j = 0;
    for (int k = 0; k < m + n; ++k) {
        if (j == n || (i < m && A[i] <= B[j])) {
            C[k] = A[i++];
        } else {
            C[k] = B[j++];
        }
    }
}

template <typename scalar_t>
__global__ void merge_sort_kernel(const scalar_t* __restrict__ src, 
                                  scalar_t* __restrict__ dst, 
                                  int N,
                                  int width) {
    int COARSE_FACTOR = width < COARSE_THREASHOLD ? COARSE_FACTOR1 : COARSE_FACTOR2;
    int k_start = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;

    if(k_start >= N) {
        return; 
    }

    int pair_width = 2 * width; 
    int pair_start = (k_start / pair_width) * pair_width; 

    const scalar_t* A = src + pair_start;
    const scalar_t* B = src + pair_start + width;

    int m = width < (N - pair_start) ? width : (N - pair_start);
    int n = 0;
    if(pair_start + width < N) {
        n = (pair_start + pair_width < N) ? width : (N - pair_start - width);
    }

    int k_end = k_start + COARSE_FACTOR < N ? k_start + COARSE_FACTOR : N;
    k_end = k_end < pair_start + m + n ? k_end : pair_start + m + n;

    int local_k_start = k_start - pair_start;
    int local_k_end = k_end - pair_start;

    int i_start = coRank<scalar_t>(A, B, m, n, local_k_start);
    int j_start = local_k_start - i_start;

    int i_end = coRank<scalar_t>(A, B, m, n, local_k_end);
    int j_end = local_k_end - i_end;

    mergeSequential<scalar_t>(&A[i_start], &B[j_start], &dst[k_start], i_end - i_start, j_end - j_start);
    
}

torch::Tensor sort_cuda(torch::Tensor data) {
    TORCH_CHECK(data.device().is_cuda(), "data must be a CUDA tensor");

    int N = torch::size(data, 0);
    int BLOCK_NUM = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    auto src = data;
    auto dst = torch::empty_like(data);

    for(int width = 1; width < N; width *= 2) {
        int COARSE_FACTOR = width < COARSE_THREASHOLD? COARSE_FACTOR1 : COARSE_FACTOR2;
        int num_blocks = (N + BLOCK_DIM * COARSE_FACTOR - 1) / (BLOCK_DIM * COARSE_FACTOR);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "merge_sort_kernel", ([&] {
            merge_sort_kernel<scalar_t><<<num_blocks, BLOCK_DIM>>>(
                src.data_ptr<scalar_t>(),
                dst.data_ptr<scalar_t>(),
                N,
                width
            );
        }));
        std::swap(src, dst);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return src;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

torch::Tensor sort_cuda(torch::Tensor data);
"""

sort_module = load_inline(
    name='sort_cuda',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda'],
    verbose=True,
)

def sort(data):
    if not data.is_cuda:
        raise ValueError("Input tensor must be a CUDA tensor")
    return sort_module.sort_cuda(data)

def custom_kernel(data: input_t) -> output_t:
    return sort(data)