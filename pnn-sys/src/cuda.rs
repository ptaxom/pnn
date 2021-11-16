use std::os::raw::c_void;

extern "C" {
    pub fn activation_mish_fp16(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t;
    pub fn activation_mish_fp32(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t;
    pub fn activation_mish_fp64(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t;
}
