# writer: @yechenzhi
# description: conv2d implementation using PyTorch with custom CUDA kernel
# algorithm: blocks same to the input tiles, then disable those threads that are out of output tiles
# results:  H100, 25124 ms
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import torch
import torch.nn.functional as F

conv2d_cuda_source = """
template <typename scalar_t>
#define IN_TILE_DIM 32
#define IN_TIME_Z 1
__global__ void conv2d_kernel(const scalar_t* __restrict__ N, 
                           const scalar_t* __restrict__ F, 
                           scalar_t* __restrict__ P,
                           int inchannels,
                           int outchannels,
                           int batch,
                           int r,
                           int in_width,
                           int in_height,
                           int out_width,
                           int out_height) {
    int in_tile_w = blockDim.x;
    int in_tile_h = blockDim.y;
    int in_tile_z = blockDim.z;
    int out_tile_w = IN_TILE_DIM - r + 1;
    int out_tile_h = IN_TILE_DIM - r + 1;

    int in_w = blockIdx.x * out_tile_w + threadIdx.x;
    int in_h = blockIdx.y * out_tile_h + threadIdx.y;
    int in_b = blockIdx.z / outchannels;
    int out_c = blockIdx.z % outchannels;
    int in_c_init = threadIdx.z;
    int get_out_times = (inchannels + in_tile_z - 1) / in_tile_z;

    scalar_t out = 0.0f;

    __shared__ scalar_t N_s[IN_TIME_Z][IN_TILE_DIM][IN_TILE_DIM];
    __shared__ scalar_t F_s[IN_TIME_Z][IN_TILE_DIM][IN_TILE_DIM];

    for(int i = 0; i < get_out_times; ++i) {
        int in_c = in_c_init + i * in_tile_z;
        if (in_w < in_width && in_h < in_height && in_b < batch && in_c < inchannels) {
            N_s[threadIdx.z][threadIdx.y][threadIdx.x] = N[in_b * inchannels * in_height * in_width +
                                                          in_c * in_height * in_width + 
                                                          in_h * in_width + 
                                                          in_w];
        } else{
            N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
        } 

        if (threadIdx.y < r && threadIdx.x < r && in_c < inchannels) {
            F_s[threadIdx.z][threadIdx.y][threadIdx.x] = F[out_c * inchannels * r * r +
                                                          in_c * r * r + 
                                                          threadIdx.y * r + 
                                                          threadIdx.x];
        } 
        else{
            F_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
        } 
        __syncthreads(); 

        if (in_w < out_width && in_h < out_height && out_c < outchannels && threadIdx.x < out_tile_w && threadIdx.y < out_tile_h) {
            for (int in_c_offset = 0; in_c_offset < in_tile_z; ++in_c_offset) {
                for (int in_h_offset = 0; in_h_offset < r; ++in_h_offset) {
                    for (int in_w_offset = 0; in_w_offset < r; ++in_w_offset) {
                        int tile_h = threadIdx.y + in_h_offset;
                        int tile_w = threadIdx.x + in_w_offset;
                        out += N_s[in_c_offset][tile_h][tile_w] * 
                               F_s[in_c_offset][in_h_offset][in_w_offset];
                    }
                }
            }
        } else {
            out += 0.0f;
        }
        __syncthreads();
    }
    if (in_w < out_width && in_h < out_height && in_b < batch && out_c < outchannels && threadIdx.x < out_tile_w && threadIdx.y < out_tile_h) { 
        int out_index = in_b * outchannels * out_height * out_width + out_c * out_height * out_width + in_h * out_width + in_w;
        P[out_index] = out;
    }
}

torch::Tensor conv2d_cuda(torch::Tensor input_tensor, torch::Tensor kernel) {
    TORCH_CHECK(input_tensor.device().is_cuda(), "input_tensor must be a CUDA tensor");
    TORCH_CHECK(kernel.device().is_cuda(), "kernel must be a CUDA tensor");
    
    int batch = torch::size(input_tensor, 0);
    int inchannels = torch::size(input_tensor, 1);
    int outchannels = torch::size(kernel, 0);
    TORCH_CHECK(inchannels == torch::size(kernel, 1), "Input channels and kernel input channels must match");
    
    int r = torch::size(kernel, 2);  // Assuming square kernel, so height == width
    int height = torch::size(input_tensor, 2);
    int width = torch::size(input_tensor, 3);
    int out_height = height - r + 1;
    int out_width = width - r + 1;
    auto P = torch::empty({batch, outchannels, out_height, out_width}, input_tensor.options()); 

    dim3 threads(IN_TILE_DIM, IN_TILE_DIM, 1);
    int out_tile_dim = IN_TILE_DIM - r + 1;
    dim3 blocks((out_width + out_tile_dim - 1) / out_tile_dim, 
                (out_height + out_tile_dim - 1) / out_tile_dim, 
                 batch * outchannels); 
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_tensor.scalar_type(), "conv2d_kernel", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input_tensor.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            P.data_ptr<scalar_t>(),
            inchannels,
            outchannels,
            batch,
            r,
            width,
            height,
            out_width,
            out_height
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return P;
}
"""

conv2d_cpp_source = """
#include <torch/extension.h>

torch::Tensor conv2d_cuda(torch::Tensor input_tensor , torch::Tensor kernel);
"""

conv2d_module = load_inline(
    name='conv2d_cuda',
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=['conv2d_cuda'],
    verbose=True,
)

def conv2d(input_tensor, kernel):
    if not input_tensor.is_cuda or not kernel.is_cuda:
        raise RuntimeError("Both tensors must be on GPU")
    return conv2d_module.conv2d_cuda(input_tensor, kernel)

def custom_kernel(data: input_t) -> output_t:
    """
    Implementation of 2D convolution using PyTorch with no padding and no striding.
    Args:
        data: Tuple of (input tensor, kernel tensor)
        spec: Convolution specifications
    Returns:
        Output tensor after convolution
    """

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    input_tensor, kernel = data
    return conv2d(
        input_tensor,
        kernel
    )