#include "kernels.h"
#include <stdio.h>

template<typename T>
__global__ void add_bias_kernel(T *inplace_data, T* biases, int n_elements, int n_channels, int channel_size) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    int bias_id = (i / channel_size) % n_channels;
    inplace_data[i] += biases[bias_id];
}

template<typename T>
cudaError_t add_bias_call(void *inplace_data, void* biases, size_t n_elements, size_t n_channels, size_t channel_size, cudaStream_t stream) {
    add_bias_kernel<T><<<get_gridsize(n_elements), BLOCK_SIZE, 0, stream >>>(
        static_cast<T*>(inplace_data),
        static_cast<T*>(biases),
        static_cast<int>(n_elements),
        static_cast<int>(n_channels),
        static_cast<int>(channel_size)
    );
    return cudaGetLastError();
}

cudaError_t add_bias(void *inplace_data, void* biases, size_t n_elements, size_t n_channels, size_t channel_size, cudnnDataType_t dtype, cudaStream_t stream) {
    if (dtype == CUDNN_DATA_HALF)
        return add_bias_call<half>(inplace_data, biases, n_elements, n_channels, channel_size, stream);
    if (dtype == CUDNN_DATA_FLOAT)
        return add_bias_call<float>(inplace_data, biases, n_elements, n_channels, channel_size, stream);
    if (dtype == CUDNN_DATA_DOUBLE)
        return add_bias_call<double>(inplace_data, biases, n_elements, n_channels, channel_size, stream);
    return cudaErrorInvalidValue;
}