# H100 pytorch 200 us
# c = 4, 53 us
# c = 16, 48.8 us
# c = 32, 41 us
# c = 64, 47 us
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

histogram_cuda_source = """
template <typename scalar_t>
#define COARSE_FACTOR 16
__global__ void histogram_kernel(const scalar_t* __restrict__ data,
                           int* __restrict__ C, 
                           int N) {
    __shared__ int C_s[256];
    if (threadIdx.x < 256) {
        C_s[threadIdx.x] = 0; 
    }
    __syncthreads();
           
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int accumulator = 0;
    int prev_bin = -1;
    for(int i = tid; i < N; i += blockDim.x * gridDim.x) {
        int value = data[i];
        if(value == prev_bin) {
            accumulator++;
        } else {
            if (accumulator > 0) {
                atomicAdd(&C_s[prev_bin], accumulator);
            }
            accumulator = 1;
            prev_bin = value;
        }
    }
    if (accumulator > 0) {
        atomicAdd(&C_s[prev_bin], accumulator);
    }
    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&C[threadIdx.x], C_s[threadIdx.x]);   
    }
}

torch::Tensor histogram_cuda(torch::Tensor data) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");
    
    int N = torch::size(data, 0); 
    auto options = torch::TensorOptions().device(data.device()).dtype(torch::kInt); 
    auto C = torch::zeros({256}, options); 

    dim3 threads(1024, 1, 1);
    dim3 blocks((N + threads.x * COARSE_FACTOR - 1) / (threads.x * COARSE_FACTOR), 1, 1);
    
    AT_DISPATCH_INTEGRAL_TYPES(data.scalar_type(), "histogram_kernel", ([&] {
        histogram_kernel<scalar_t><<<blocks, threads>>>(
            data.data_ptr<scalar_t>(),
            C.data_ptr<int>(),
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

histogram_cpp_source = """
#include <torch/extension.h>

torch::Tensor histogram_cuda(torch::Tensor data);
"""

histogram_module = load_inline(
    name='histogram_cuda',
    cpp_sources=histogram_cpp_source,
    cuda_sources=histogram_cuda_source,
    functions=['histogram_cuda'],
    verbose=True,
)

def histogram(data):
    if not data.is_cuda:
        raise RuntimeError("Both tensors must be on GPU")
    return histogram_module.histogram_cuda(data)

def custom_kernel(data: input_t) -> output_t:
    return histogram(data)
