#include "kernels.h"

template<typename T>
__device__ T mish(T x) {
    T e = exp(x);
    T n = e * e + 2 * e;
    if (x <= -0.6f)
        return x * n / (n + 2);
    return x - 2 * x / (n + 2);
}

template<>
__device__ __half mish(__half x) {
    __half e = hexp(x);
    half HALF_TWO = __float2half(2.f);
    half n = __hadd(__hmul(e, e), __hmul(e, HALF_TWO));
    half n2 = __hadd(n, HALF_TWO);
    if (__hle(x, -0.6f))
        return __hmul(x , __hdiv(n, n2));
    return __hsub(x, __hmul(2.f , __hdiv(x, n2)));
}

template<>
__device__ float mish(float x) {
    float e = __expf(x);
    float n = e * e + 2 * e;
    if (x <= -0.6f)
        return x * __fdividef(n, n + 2);
    return x - 2 * __fdividef(x, n + 2);
}

template<typename T>
__global__ void activation_mish_kernel(T* data, int elements) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < elements) {
        data[i] = mish<T>(data[i]);
    }
}

template<typename T>
cudaError_t activation_mish(void* data, size_t elements, cudaStream_t stream) {
    activation_mish_kernel<T><<<get_gridsize(elements), BLOCK_SIZE, 0, stream >>>(
        static_cast<T*>(data), 
        static_cast<int>(elements)
    );
    return cudaGetLastError();
}

cudaError_t activation_mish_fp16(void* data, size_t elements, cudaStream_t stream){
    return activation_mish<__half>(data, elements, stream);
}

cudaError_t activation_mish_fp32(void* data, size_t elements, cudaStream_t stream){
    return activation_mish<float>(data, elements, stream);
}

cudaError_t activation_mish_fp64(void* data, size_t elements, cudaStream_t stream) {
    return activation_mish<double>(data, elements, stream);
}