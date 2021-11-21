#include "kernels.h"
#include <cudnn.h>

// template magic not work here :(
__global__ void convert_kernel_half2float(float* output, half* input, int n_elements) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        output[i] = __half2float(input[i]);
    }
}

__global__ void convert_kernel_float2half(half* output, float* input, int n_elements) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        output[i] = __half2float(input[i]);
    }
}

cudaError_t cvt_ptr_data(void* output, void* input, size_t n_elements, size_t otype, size_t itype, cudaStream_t stream) {
    if (otype == itype)
        return cudaSuccess;

    if (otype == CUDNN_DATA_FLOAT && itype == CUDNN_DATA_HALF) {
        convert_kernel_half2float<<<get_gridsize(n_elements), BLOCK_SIZE, 0, stream >>>(
            static_cast<float*>(output), 
            static_cast<half*>(input), 
            static_cast<int>(n_elements)
        );
        return cudaGetLastError();
    }

    if (otype == CUDNN_DATA_HALF && itype == CUDNN_DATA_FLOAT){
        convert_kernel_float2half<<<get_gridsize(n_elements), BLOCK_SIZE, 0, stream >>>(
            static_cast<half*>(output), 
            static_cast<float*>(input), 
            static_cast<int>(n_elements)
        );
        return cudaGetLastError();
    }
    
    return cudaErrorInvalidValue;
    
}