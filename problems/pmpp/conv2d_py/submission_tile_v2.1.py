# writer: @yechenzhi
# description: conv2d implementation using PyTorch with custom CUDA kernel
# algorithm: blocks same to the output tiles, then each thread load all the input tiles
# results:  H100, 192 ms
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import torch
import torch.nn.functional as F

conv2d_cuda_source = """
template <typename scalar_t>
#define OUT_TILE_DIM 32
#define IN_TILE_DIM 64
#define MAX_KERNEL_DIM 32
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
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;
    int in_c_init = threadIdx.z;
    int b = z / outchannels;
    int out_c = z % outchannels;

    int in_tile_width = OUT_TILE_DIM + r - 1;
    int in_tile_height = OUT_TILE_DIM + r - 1;
    int in_tile_z = IN_TIME_Z;

    int get_out_times = (inchannels + in_tile_z - 1) / in_tile_z;

    scalar_t out = 0.0f;

    __shared__ scalar_t N_s[IN_TIME_Z][IN_TILE_DIM][IN_TILE_DIM];
    __shared__ scalar_t F_s[IN_TIME_Z][MAX_KERNEL_DIM][MAX_KERNEL_DIM];

    for(int i = 0; i < get_out_times; ++i) {
        int in_c = in_c_init + i * in_tile_z;
        int tile_start_row = blockDim.y * blockIdx.y;
        int tile_start_col = blockDim.x * blockIdx.x;
        
        const scalar_t* N_ptr = &N[b * inchannels * in_height * in_width +
                                   in_c * in_height * in_width];
        
        if(in_c < inchannels) {
            int num_threads_2d = blockDim.x * blockDim.y;
            int thread_id_2d = threadIdx.y * blockDim.x + threadIdx.x;

            for(int j = thread_id_2d; j < in_tile_width * in_tile_height; j += num_threads_2d) {
                int dest_row = j / in_tile_width;
                int dest_col = j % in_tile_width;

                int src_row = tile_start_row + dest_row;
                int src_col = tile_start_col + dest_col;

                if (src_row < in_height && src_col < in_width) {
                    N_s[threadIdx.z][dest_row][dest_col] = N_ptr[src_row * in_width + src_col];
                } else {
                    N_s[threadIdx.z][dest_row][dest_col] = 0.0f;
                }
            }
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

        if (col < out_width && row < out_height && z < outchannels * batch){
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
        }
        
        __syncthreads();
    }
    if (col < out_width && row < out_height && z < outchannels * batch) { 
        int out_index = b * outchannels * out_height * out_width +
                        out_c * out_height * out_width +
                        row * out_width + col;      
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

    dim3 threads(OUT_TILE_DIM, OUT_TILE_DIM, IN_TIME_Z);
    dim3 blocks((out_width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 
                (out_height + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 
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