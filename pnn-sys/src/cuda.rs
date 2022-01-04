use std::os::raw::{c_void, c_int, c_char};

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct InferStats {
    inference_time: f64,
    inference_with_nms: f64,
    pub total_time: f64,
    total_frames: usize,
}

impl InferStats {
    pub fn fps(&self) -> f64 {
        (self.total_frames as f64) / (self.total_time / 1000.)
    }

    pub fn infer_fps(&self, batchsize: usize) -> f64 {
        (self.total_frames as f64) / (self.inference_time / 1000. / batchsize as f64)
    }

    pub fn nms_fps(&self, batchsize: usize) -> f64 {
        (self.total_frames as f64) / (self.inference_with_nms / 1000. / batchsize as f64)
    }

    pub fn total_frames(&self) -> usize {
        self.total_frames
    }
}

extern "C" {
    pub fn activation_mish_fp16(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t;
    pub fn activation_mish_fp32(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t;
    pub fn activation_mish_fp64(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t;

    pub fn upsample_forward_fp16(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t;
    pub fn upsample_forward_fp32(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t;
    pub fn upsample_forward_fp64(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t;

    pub fn add_bias(inplace_data: *mut c_void, biases: *mut c_void, n_elements: usize, n_channels: usize, channel_size: usize, dtype: usize, stream: cudaStream_t) -> cudaError_t;

    pub fn load_image2batch(image_path: *const c_char, batch_id: usize, width: usize,  height: usize, input_data: *mut c_void) -> c_int;

    pub fn cvt_ptr_data(output: *mut c_void, input: *mut c_void, n_elements: usize, otype: usize, itype: usize, stream: cudaStream_t) -> cudaError_t;
    pub fn render_bboxes(image_path: *const c_char, n_boxes: usize, boxes: *const c_void, classes: *const *const c_char, window_name: *const c_char) -> c_int;

    pub fn visual_demo(video_path:  *const c_char,
        classes: *const *const c_char,
        batchsize: usize,
        width: usize,
        height: usize,
        inp_ptr: *mut c_void,
        model_ptr: *mut c_void,
        infer_call: extern fn(*mut c_void, *mut usize, *mut f64) -> *mut c_void,
        output: *const c_char,
        show: bool
        ) -> InferStats;
}