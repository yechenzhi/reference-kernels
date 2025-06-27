# pytorch H100: 315 us
# triton H100: 325 us
# CUDA H100: COARSE=1, 190 us
# CUDA H100: COARSE=8, 102 us
# CUDA H100: COARSE=16, 102 us

# CUDA shuf: 99 us
 

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

vectorsum_cuda_source = """
template <typename scalar_t>
#define BLOCK_DIM 1024
#define COARSE_FACTOR 4
#define WARP_SIZE 32
__global__ void vectorsum_kernel(const scalar_t* __restrict__ data, 
                            float* __restrict__ output,
                            int N) {
    int segment = COARSE_FACTOR * blockIdx.x * blockDim.x * 2;
    int idx = segment + threadIdx.x;
    
    __shared__ scalar_t data_s[BLOCK_DIM];
    float threadSum = 0.0f;
    for (int c = 0; c < 2 * COARSE_FACTOR; ++c) {
        if (idx + c * blockDim.x < N){
            threadSum += data[idx + c * blockDim.x];
        }
    }
    data_s[threadIdx.x] = threadSum;
    __syncthreads();

    for (int stride = BLOCK_DIM / 2; stride > WARP_SIZE; stride /= 2) {
        if (threadIdx.x < stride) {
            data_s[threadIdx.x] += data_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float sum = 0.0f;
    if (threadIdx.x < WARP_SIZE) {
        sum = data_s[threadIdx.x] + data_s[threadIdx.x + WARP_SIZE];

        for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, stride);
        }
    } 

    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}


torch::Tensor vectorsum_cuda(torch::Tensor data) {
    TORCH_CHECK(data.device().is_cuda(), "data must be a CUDA tensor");

    int N = torch::size(data, 0);
    int BLOCK_NUM = (N + BLOCK_DIM * 2 * COARSE_FACTOR - 1) / (BLOCK_DIM * 2 * COARSE_FACTOR);

    auto output = torch::zeros({}, data.options().dtype(torch::kFloat32));
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "vectorsum_kernel", ([&] {
        vectorsum_kernel<scalar_t><<<BLOCK_NUM, BLOCK_DIM>>>(
            data.data_ptr<scalar_t>(),
            output.data_ptr<float>(),
            N
        );
    }));


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output.to(data.scalar_type());
}
"""

vectorsum_cpp_source = """
#include <torch/extension.h>

torch::Tensor vectorsum_cuda(torch::Tensor data);
"""

vectorsum_module = load_inline(
    name='vectorsum_cuda',
    cpp_sources=vectorsum_cpp_source,
    cuda_sources=vectorsum_cuda_source,
    functions=['vectorsum_cuda'],
    verbose=True,
)

def vectorsum(data):
    if not data.is_cuda:
        raise RuntimeError("Both tensors must be on GPU")
    return vectorsum_module.vectorsum_cuda(data)

def custom_kernel(data: input_t) -> output_t:
    return vectorsum(data)