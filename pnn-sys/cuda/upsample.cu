#include "kernels.h"

template<typename T>
__device__ void scale_value(T scale, T *in, T *out, int out_index, int in_index) {
    out[out_index] += scale * in[in_index];
}

template<>
__device__ void scale_value(half scale, half *in, half *out, int out_index, int in_index) {
    out[out_index] =  __hadd(__hmul(scale, in[in_index]), out[out_index]);
}


// TODO: add optimized version using shared memory
template<typename T>
__global__ void upsample_forward_kernel(size_t elements, T* input, int n, int c, int h, int w, int stride, T scale, T* output) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    int out_index = i;

    int out_w = i % (w*stride);

    i = i / (w * stride);
    int out_h = i % (h*stride);

    i = i / (h * stride);
    int out_c = i % c;

    i = i / c;
    int b = i % n;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b * w * h * c + in_c * w * h + in_h * w + in_w;
    scale_value(scale, input, output, out_index, in_index);    
}

template<typename T>
cudaError_t upsample_forward(void* input, size_t n, size_t c, size_t h, size_t w, size_t stride, float scale, void* output, cudaStream_t stream) {
    size_t elements = n * c * h * w * stride * stride;
    upsample_forward_kernel<T><<<get_gridsize(elements), BLOCK_SIZE, 0, stream >>>(
        elements,
        static_cast<T*>(input), 
        static_cast<int>(n),
        static_cast<int>(c),
        static_cast<int>(h),
        static_cast<int>(w),
        static_cast<int>(stride),
        static_cast<T>(scale),
        static_cast<T*>(output)
    );
    return cudaGetLastError();
}

cudaError_t upsample_forward_fp16(void* input, size_t n, size_t c, size_t h, size_t w, size_t stride, float scale, void* output, cudaStream_t stream) {
    return upsample_forward<half>(input, n, c, h, w, stride, scale, output, stream);
}

cudaError_t upsample_forward_fp32(void* input, size_t n, size_t c, size_t h, size_t w, size_t stride, float scale, void* output, cudaStream_t stream) {
    return upsample_forward<float>(input, n, c, h, w, stride, scale, output, stream);
}

cudaError_t upsample_forward_fp64(void* input, size_t n, size_t c, size_t h, size_t w, size_t stride, float scale, void* output, cudaStream_t stream) {
    return upsample_forward<double>(input, n, c, h, w, stride, scale, output, stream);
}