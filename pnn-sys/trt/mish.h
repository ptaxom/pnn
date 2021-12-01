#pragma once

#include <cuda_runtime.h>

cudaError_t trt_activation_mish_fp32(cudaStream_t stream, size_t elements, const void* input, void* output);

cudaError_t trt_activation_mish_fp16(cudaStream_t stream, size_t elements, const void* input, void* output);