#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

const size_t BLOCK_SIZE = 512;


extern "C" {
    dim3 get_gridsize(size_t elements);

    cudaError_t activation_mish_fp16(void* data, size_t elements, cudaStream_t stream);
    cudaError_t activation_mish_fp32(void* data, size_t elements, cudaStream_t stream);
    cudaError_t activation_mish_fp64(void* data, size_t elements, cudaStream_t stream);

    cudaError_t upsample_forward_fp16(void* input, size_t n, size_t c, size_t h, size_t w, size_t stride, float scale, void* output, cudaStream_t stream);
    cudaError_t upsample_forward_fp32(void* input, size_t n, size_t c, size_t h, size_t w, size_t stride, float scale, void* output, cudaStream_t stream);
    cudaError_t upsample_forward_fp64(void* input, size_t n, size_t c, size_t h, size_t w, size_t stride, float scale, void* output, cudaStream_t stream);

}