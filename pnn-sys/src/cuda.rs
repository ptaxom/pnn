use std::os::raw::c_void;

extern "C" {
    pub fn activation_mish_fp16(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t;
    pub fn activation_mish_fp32(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t;
    pub fn activation_mish_fp64(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t;

    pub fn upsample_forward_fp16(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t;
    pub fn upsample_forward_fp32(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t;
    pub fn upsample_forward_fp64(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t;
}
