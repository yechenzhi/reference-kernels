# results:  H100, 294 ms
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import torch
import torch.nn.functional as F

conv2d_cuda_source = """
template <typename scalar_t>
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
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int z_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_w < out_width && out_h < out_height && z_idx < outchannels * batch){
        int out_b = z_idx / outchannels;
        int out_c = z_idx % outchannels;
        scalar_t sum = 0;
        for(int in_c = 0; in_c < inchannels; ++in_c) {
            for(int in_h_offset = 0; in_h_offset < r; ++in_h_offset) {
                for(int in_w_offset = 0; in_w_offset < r; ++in_w_offset){
                    int in_h = out_h + in_h_offset;
                    int in_w = out_w + in_w_offset;
                    if (in_h < in_height && in_w < in_width){
                        int in_index = out_b * inchannels * in_height * in_width + 
                                   in_c * in_height * in_width  + 
                                   in_h * in_width + 
                                   in_w;
                        int filter_index = out_c * inchannels * r * r + 
                                       in_c * r * r + 
                                       in_h_offset * r + 
                                       in_w_offset;
                        sum += N[in_index] * F[filter_index];
                    }
                }
            }
        }
        int out_index = out_b * outchannels * out_height * out_width + 
                        out_c * out_height * out_width + 
                        out_h * out_width + 
                        out_w;
        P[out_index] = sum;
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

    dim3 threads(32, 32, 1);
    dim3 blocks((out_width + threads.x - 1) / threads.x, 
                (out_height + threads.y - 1) / threads.y, 
                (batch * outchannels + threads.z - 1) / threads.z); 
    
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