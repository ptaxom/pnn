use std::os::raw::{c_void, c_int, c_char};

extern "C" {
    pub fn activation_mish_fp16(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t;
    pub fn activation_mish_fp32(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t;
    pub fn activation_mish_fp64(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t;

    pub fn upsample_forward_fp16(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t;
    pub fn upsample_forward_fp32(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t;
    pub fn upsample_forward_fp64(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t;

    pub fn load_image2batch(image_path: *const c_char, batch_id: usize, width: usize,  height: usize, input_data: *mut c_void) -> c_int;

    pub fn cvt_ptr_data(output: *mut c_void, input: *mut c_void, n_elements: usize, otype: usize, itype: usize, stream: cudaStream_t) -> cudaError_t;
    pub fn render_bboxes(image_path: *const c_char, n_boxes: usize, boxes: *const c_void, classes: *const *const c_char, window_name: *const c_char) -> c_int;
}