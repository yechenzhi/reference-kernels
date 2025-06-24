# pytorch H100: 1037 us
# CUDA H100: 3.2 ms
# CUDA H100 double buffer: 2.8 ms
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cumsum_cuda_source = """
template <typename scalar_t>
#define BLOCK_DIM 1024
__global__ void cumsum_kernel(const scalar_t* __restrict__ data, 
                            scalar_t* __restrict__ output,
                            scalar_t* __restrict__ partialSums,
                            int N) {
    __shared__ scalar_t buffer1_s[BLOCK_DIM];
    __shared__ scalar_t buffer2_s[BLOCK_DIM];
    scalar_t* inBuffer_s = buffer1_s;
    scalar_t* outBuffer_s = buffer2_s;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {  
        inBuffer_s[threadIdx.x] = data[idx];
    } else {
        inBuffer_s[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for(int stride = 1; stride < BLOCK_DIM; stride *= 2) {
        if(threadIdx.x >= stride) {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        scalar_t* tmp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = tmp;
    }
    if (threadIdx.x == BLOCK_DIM - 1) {
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x];
    }
    if (idx < N){
        output[idx] = inBuffer_s[threadIdx.x];
    }
}

template <typename scalar_t>
__global__ void add_kernel(scalar_t* __restrict__ output, 
                           const scalar_t* __restrict__ partialSums, 
                           int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && idx < N) {
        output[idx] += partialSums[blockIdx.x - 1]; 
    }
}

torch::Tensor cumsum_cuda(torch::Tensor data) {
    TORCH_CHECK(data.device().is_cuda(), "data must be a CUDA tensor");

    int N = torch::size(data, 0);
    int BLOCK_NUM = (N + BLOCK_DIM - 1) / BLOCK_DIM;

    auto partialSums = torch::zeros({BLOCK_NUM}, data.options());
    auto output = torch::empty_like(data);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "cumsum_kernel", ([&] {
        cumsum_kernel<scalar_t><<<BLOCK_NUM, BLOCK_DIM>>>(
            data.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            partialSums.data_ptr<scalar_t>(),
            N
        );
    }));

    if (BLOCK_NUM > 1) {
        partialSums = cumsum_cuda(partialSums);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "add_kernel", ([&] {
            add_kernel<scalar_t><<<BLOCK_NUM, BLOCK_DIM>>>(
                output.data_ptr<scalar_t>(),
                partialSums.data_ptr<scalar_t>(),
                N
            );
        }));
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

cumsum_cpp_source = """
#include <torch/extension.h>

torch::Tensor cumsum_cuda(torch::Tensor data);
"""

cumsum_module = load_inline(
    name='cumsum_cuda',
    cpp_sources=cumsum_cpp_source,
    cuda_sources=cumsum_cuda_source,
    functions=['cumsum_cuda'],
    verbose=True,
)

def cumsum(data):
    if not data.is_cuda:
        raise RuntimeError("Both tensors must be on GPU")
    return cumsum_module.cumsum_cuda(data)

def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of inclusive prefix sum using PyTorch.
    Args:
        data: Input tensor to compute prefix sum on
    Returns:
        Tensor containing the inclusive prefix sum
    """
    return cumsum(data)