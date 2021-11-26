use crate::nn::{BoundingBox, BuildError, Network};
use crate::cudnn::cudnnDataType;
use std::os::raw::{c_void, c_int, c_char};
extern crate libc;
use std::mem;

extern "C" fn infer_call(net: *mut c_void,n_boxes: *mut usize) -> *mut c_void {
    unsafe {
        let network: *mut Network = net as *mut Network;
        (*network).forward().unwrap();

        let mut total_boxes: usize = 0;
        let predictions = (*network).get_yolo_predictions().unwrap();
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

pub fn demo(video_path: &String, 
            config_file: &String,
            weight_path: &String,
            classes_file: &String, 
            data_type: &cudnnDataType,
            batchsize: usize,
            threshold: f32, 
            nms_threshold: f32
    ) -> Result<(), BuildError> {
        let classes = crate::parsers::load_classes(classes_file).map_err( |e| {
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

        let mut net = crate::nn::Network::from_darknet(config_file)?;
        net.set_batchsize(batchsize)?;
        net.load_darknet_weights(weight_path)?;
        net.build(data_type)?;
        println!("Builded yolo");
        net.set_detections_ops(threshold, nms_threshold);
        
        unsafe {
            pnn_sys::visual_demo(c_video_path.as_ptr(),
            ffi_ptrs.as_mut_ptr(),
                batchsize,
                net.get_size().1,
                net.get_size().0,
                net.get_input_ptr()?,
                std::ptr::addr_of!(net) as *mut c_void,
                infer_call
                );
        }
    Ok(())
}