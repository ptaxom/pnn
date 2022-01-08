#include "kernels.h"

template<typename T>
__device__ T leaky(T x) {
    if (x <= 0)
        return x * .1f;
    return x;
}

template<>
__device__ __half leaky(__half x) {
    if (__hle(x, 0.))
        return __hmul(x, .1f);
    return x;
}

template<typename T>
__global__ void activation_leaky_kernel(T* data, int elements) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < elements) {
        data[i] = leaky<T>(data[i]);
    }
}

template<typename T>
cudaError_t activation_leaky(void* data, size_t elements, cudaStream_t stream) {
    activation_leaky_kernel<T><<<get_gridsize(elements), BLOCK_SIZE, 0, stream >>>(
        static_cast<T*>(data), 
        static_cast<int>(elements)
    );
    return cudaGetLastError();
}

cudaError_t activation_leaky_fp16(void* data, size_t elements, cudaStream_t stream){
    return activation_leaky<__half>(data, elements, stream);
}

cudaError_t activation_leaky_fp32(void* data, size_t elements, cudaStream_t stream){
    return activation_leaky<float>(data, elements, stream);
}

cudaError_t activation_leaky_fp64(void* data, size_t elements, cudaStream_t stream) {
    return activation_leaky<double>(data, elements, stream);
}