#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

const size_t BLOCK_SIZE = 512;

extern "C" {

    cudaError_t activation_mish_fp16(void* data, size_t elements, cudaStream_t stream);
    cudaError_t activation_mish_fp32(void* data, size_t elements, cudaStream_t stream);
    cudaError_t activation_mish_fp64(void* data, size_t elements, cudaStream_t stream);
}