use crate::nn::{BoundingBox, BuildError, Network};
use crate::cudnn::cudnnDataType;
use std::os::raw::{c_void};
use std::time::{Instant};
extern crate libc;
use std::mem;

extern "C" fn infer_call(net: *mut c_void,n_boxes: *mut usize, infer_time: *mut f64) -> *mut c_void {
    unsafe {
        let network: *mut Network = net as *mut Network;
        {
            let now = Instant::now();
            (*network).forward().unwrap();
            *infer_time = now.elapsed().as_secs_f64();
        }

        let mut total_boxes: usize = 0;
        let predictions = (*network).get_detections().unwrap();
        for i in 0..predictions.len() {
            let sample_bboxes = predictions[i].len();
            *n_boxes = sample_bboxes;
            total_boxes += sample_bboxes;
        }

        let mut boxes: *mut BoundingBox = libc::malloc(mem::size_of::<BoundingBox>() * (1 + total_boxes)) as *mut BoundingBox;

        if boxes.is_null() {
            panic!("failed to allocate memory");
        }

        let origin = boxes.clone();
        for i in 0..predictions.len() {
            for bbox in &predictions[i] {
                *boxes = bbox.clone();
                boxes = boxes.offset(1);
            }
        }
        origin as *mut c_void 
    }
}

pub fn demo(video_path: String, 
            config_file: String,
            weight_path: String,
            classes_file: String, 
            half: bool,
            batchsize: usize,
            threshold: f32, 
            nms_threshold: f32,
            show: bool,
            output: Option<String>,
            trt: bool
    ) -> Result<(), BuildError> {
        let classes = crate::parsers::load_classes(&classes_file).map_err( |e| {
            BuildError::Io(e)
        })?;

        let c_video_path = std::ffi::CString::new(video_path.clone()).unwrap();
        let ffi_classes: Vec<std::ffi::CString> = classes.iter().map(|x| {
            std::ffi::CString::new(x.as_str()).unwrap()
        }).collect();
        let mut ffi_ptrs: Vec<*const std::os::raw::c_char> = ffi_classes.iter().map(|x| {
            x.as_ptr()
        }).collect();
        ffi_ptrs.push(std::ptr::null());

        let mut data_type = cudnnDataType::FLOAT;
        let mut net = crate::nn::Network::from_darknet(&config_file)?;
        if !trt {
            data_type = if half {cudnnDataType::HALF} else {cudnnDataType::FLOAT};
            net.build_cudnn(batchsize, data_type.clone(), Some(weight_path.clone()))?;
        } else {
            net.load_trt(&weight_path)?;
        }
        // Warm-up
        for _ in 0..5 {
            net.forward().map_err(|e| {
                BuildError::Runtime(e)
            })?;
        }
        let i_ptr = net.get_input_ptr().borrow_mut().ptr() as *mut std::os::raw::c_void;
        println!("Loaded yolo engine");
        net.set_detections_ops(threshold, nms_threshold);
        let c_output_isnull = output.is_none();
        let c_output_ptr = std::ffi::CString::new(output.unwrap_or(String::from("default.avi"))).unwrap();
        
        let stats;
        unsafe {
            stats = pnn_sys::visual_demo(c_video_path.as_ptr(),
                ffi_ptrs.as_mut_ptr(),
                batchsize,
                net.get_size().1,
                net.get_size().0,
                i_ptr,
                std::ptr::addr_of!(net) as *mut c_void,
                infer_call,
                if c_output_isnull {std::ptr::null()} else {c_output_ptr.as_ptr()},
                show
                );
        }
    println!("Stats for      {}", video_path);
    println!("Data type:     {}", data_type);
    println!("Batchsize:     {}", batchsize);
    println!("Total frames:  {}", stats.total_frames());
    println!("FPS:           {}", stats.fps());
    println!("INF+NMS FPS:   {}", stats.nms_fps(batchsize));
    println!("Inference FPS: {}", stats.infer_fps(batchsize));
    Ok(())
}