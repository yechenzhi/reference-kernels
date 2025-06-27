# writer: @yechenzhi
# description: conv2d implementation using PyTorch with custom CUDA kernel
# algorithm: blocks same to the output tiles, then each thread load all the input tiles, and there are batch*inChannels*outChannels blocks
# results:  H100, ? ms
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import torch
import torch.nn.functional as F

conv2d_cuda_source = """
#include <cuda_fp16.h>

// --- Kernel Configuration Defines ---
// The dimensions of the tile processed by each thread block in the output tensor.
#define OUT_TILE_DIM 16
// The maximum supported dimension of the convolution kernel (filter).
#define MAX_KERNEL_DIM 32
// The number of threads in the Z-dimension of a block, used for parallelizing over channels.
#define IN_TIME_Z 4

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

    // --- Corrected & Simplified Indexing Logic ---
    // The grid is flattened across (batch, out_channels, in_channels).
    // Each thread gets a unique global index 'z_global_idx' corresponding to one input channel's contribution.
    int z_global_idx = blockIdx.z * blockDim.z + threadIdx.z;

    // Boundary check: Exit if the thread is outside the problem domain.
    // This is necessary because the grid size is rounded up.
    if (z_global_idx >= batch * outchannels * inchannels) {
        return;
    }

    // De-flatten the global Z index to get batch, output channel, and input channel indices.
    // This mapping is now robust and correct.
    const int in_c = z_global_idx % inchannels;
    const int temp_idx = z_global_idx / inchannels;
    const int out_c = temp_idx % outchannels;
    const int b = temp_idx / outchannels;

    // --- Output Pixel Coordinate Calculation ---
    // Each thread in the XY-plane of the block is responsible for one output pixel.
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // --- Shared Memory Declaration ---
    // The input tile in shared memory must be larger than the output tile 
    // to accommodate the kernel's footprint (halo).
    // Its size is (OUT_TILE_DIM + r - 1).
    const int IN_TILE_SIZE = OUT_TILE_DIM + r - 1;
    __shared__ scalar_t N_s[IN_TIME_Z][OUT_TILE_DIM + MAX_KERNEL_DIM - 1][OUT_TILE_DIM + MAX_KERNEL_DIM - 1];
    __shared__ scalar_t F_s[IN_TIME_Z][MAX_KERNEL_DIM][MAX_KERNEL_DIM];

    // --- Corrected Input Tile Loading ---
    // Threads in a block cooperatively load a contiguous input tile into shared memory.
    const scalar_t* N_base_ptr = &N[b * inchannels * in_height * in_width + in_c * in_height * in_width];
    int in_tile_x_base = blockIdx.x * OUT_TILE_DIM;
    int in_tile_y_base = blockIdx.y * OUT_TILE_DIM;

    // A strided loop is used because the input tile (e.g., 18x18 for r=3) is larger 
    // than the thread block's XY dimensions (16x16).
    for (int i = threadIdx.y; i < IN_TILE_SIZE; i += blockDim.y) {
        for (int j = threadIdx.x; j < IN_TILE_SIZE; j += blockDim.x) {
            int load_y = in_tile_y_base + i;
            int load_x = in_tile_x_base + j;

            // Load data if within bounds, otherwise pad with zero.
            if (load_y < in_height && load_x < in_width) {
                N_s[threadIdx.z][i][j] = N_base_ptr[load_y * in_width + load_x];
            } else {
                N_s[threadIdx.z][i][j] = 0.0f;
            }
        }
    }

    // --- Filter Loading ---
    // Each thread in the Z-dimension loads its corresponding filter.
    // Threads in the XY-plane cooperate to load the r*r filter weights.
    int num_threads_2d = blockDim.x * blockDim.y;
    int thread_id_2d = threadIdx.y * blockDim.x + threadIdx.x;
    const scalar_t* F_ptr = &F[out_c * inchannels * r * r + in_c * r * r];

    for (int i = thread_id_2d; i < r * r; i += num_threads_2d) {
        int kernel_row = i / r;
        int kernel_col = i % r;
        F_s[threadIdx.z][kernel_row][kernel_col] = F_ptr[i];
    }
    
    __syncthreads(); 

    // --- Corrected Convolution Calculation ---
    scalar_t out = 0.0f;
    // Check if the thread's output pixel is within the output tensor's bounds.
    if (col < out_width && row < out_height){
        // Perform the 2D convolution using data from shared memory.
        // There is no loop over input channels here; that reduction is handled by atomicAdd across blocks.
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < r; ++j) {
                out += N_s[threadIdx.z][threadIdx.y + i][threadIdx.x + j] * F_s[threadIdx.z][i][j];
            }
        }
    }
    
    // --- Atomic Add to Output Tensor ---
    // Atomically add the calculated partial sum to the final output tensor.
    // This correctly sums the contributions from all input channels.
    if (col < out_width && row < out_height && out != 0.0f) { 
        int out_index = b * outchannels * out_height * out_width +
                        out_c * out_height * out_width +
                        row * out_width + col;      
        atomicAdd((float*)&P[out_index], static_cast<float>(out));
    }
}

torch::Tensor conv2d_cuda(torch::Tensor input_tensor, torch::Tensor kernel) {
    TORCH_CHECK(input_tensor.device().is_cuda(), "input_tensor must be a CUDA tensor");
    TORCH_CHECK(kernel.device().is_cuda(), "kernel must be a CUDA tensor");
    
    int batch = input_tensor.size(0);
    int inchannels = input_tensor.size(1);
    int in_height = input_tensor.size(2);
    int in_width = input_tensor.size(3);

    int outchannels = kernel.size(0);
    TORCH_CHECK(inchannels == kernel.size(1), "Input channels and kernel input channels must match");
    
    int r = kernel.size(2);
    TORCH_CHECK(r <= MAX_KERNEL_DIM, "Kernel size exceeds MAX_KERNEL_DIM");

    int out_height = in_height - r + 1;
    int out_width = in_width - r + 1;
    auto P = torch::zeros({batch, outchannels, out_height, out_width}, input_tensor.options()); 

    dim3 threads(OUT_TILE_DIM, OUT_TILE_DIM, IN_TIME_Z);
    
    // The grid size calculation for the Z-dimension is based on the total number of 
    // (b, out_c, in_c) combinations, divided by the number of threads in the Z-dimension of a block.
    dim3 blocks((out_width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 
                (out_height + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 
                (batch * outchannels * inchannels + IN_TIME_Z - 1) / IN_TIME_Z); 
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_tensor.scalar_type(), "conv2d_kernel", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input_tensor.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            P.data_ptr<scalar_t>(),
            inchannels,
            outchannels,
            batch,
            r,
            in_width,
            in_height,
            out_width,
            out_height
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // This will now provide a more meaningful error message from CUDA.
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return P;
}
"""

conv2d_cpp_source = """
#include <torch/extension.h>

// Forward declaration of the CUDA function
torch::Tensor conv2d_cuda(torch::Tensor input_tensor , torch::Tensor kernel);
"""

# JIT compilation of the C++/CUDA code
conv2d_module = load_inline(
    name='conv2d_cuda',
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=['conv2d_cuda'],
    verbose=True
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
    Returns:
        Output tensor after convolution
    """

    # It's good practice to disable TF32 for precision-sensitive tests or comparisons
    torch.backends.cudnn.allow_tf32 = False
    # Deterministic algorithms can help in debugging
    torch.backends.cudnn.deterministic = True
    
    input_tensor, kernel = data
    return conv2d(
        input_tensor,
        kernel
    )