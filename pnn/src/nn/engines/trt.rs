use crate::cudnn::{cudnnDataType};
use crate::nn::{DetectionsParser, BoundingBox, RuntimeError, BuildError, Shape, ActivationType};
use crate::cudnn::{DevicePtr};
use crate::nn::Engine;

use std::{
    rc::Rc,
    cell::RefCell,
    collections::HashMap,
    os::raw::{c_void}
};


type Bindings = HashMap<String, DevMemory>;
type DevMemory = Rc<RefCell<DevicePtr>>;

pub struct TRTBuilder {
    // datatype
    data_type: cudnnDataType,
    // Net size, can be usefull for postprocess
    input_size: (usize, usize),
    // Batchsize
    batchsize: usize,
    // Inference operations
    layer_indeces: Vec<usize>,
    // TRTBuilder itself
    builder: *mut c_void
}

fn get_error<T>(msg: &str) -> Result<T, BuildError> {
    Err(BuildError::Runtime(RuntimeError::Other(String::from(msg))))
}

impl TRTBuilder {

    pub fn new(data_type: cudnnDataType, batchsize: usize, input_size: (usize, usize)) -> Result<TRTBuilder, BuildError> {
        let layer_indeces = Vec::new();
        let mut builder = std::ptr::null_mut();
        unsafe {
            builder = pnn_sys::builder_create(data_type.clone() as usize, batchsize);
        }
        if builder == std::ptr::null_mut() {
            return get_error("Couldnt create TRTBuilder")
        }

        Ok(TRTBuilder{batchsize, input_size, data_type, layer_indeces, builder})
    }

    pub fn add_input(&mut self, name: &String, shape: Rc<dyn Shape>) -> Result<usize, BuildError> {
        let cname = std::ffi::CString::new(name.clone()).unwrap();
        unsafe {
            let id = pnn_sys::builder_add_input(self.builder, 
                cname.as_ptr(), 
                shape.C(),
                shape.H().unwrap(),
                shape.W().unwrap()
            );
            if id < 0 {
                return get_error("Could add Input to TRTBuilder")
            }
            self.layer_indeces.push(id as usize);
            Ok(id as usize)
        }
    }

    pub fn add_yolo(&mut self, ind: usize, name: &String) -> Result<(), BuildError> {
        let cname = std::ffi::CString::new(name.clone()).unwrap();
        unsafe {
             pnn_sys::builder_add_yolo(self.builder, 
                ind,
                cname.as_ptr()
            );
        }
        Ok(())
    }

    pub fn add_convolution(&mut self,
        ind: usize,
        filters: usize, 
        input_channels: usize,
        kernel_size: usize,
        kernels: &Vec<f32>,
        biases: &Vec<f32>
    ) -> Result<usize, BuildError> { 
        unsafe {
            let id = pnn_sys::builder_add_convolution(
                self.builder,
                ind,
                filters,
                input_channels,
                kernel_size,
                kernels.as_ptr(),
                biases.as_ptr()
            );
            if id < 0 {
                return get_error("Could add Convolution to TRTBuilder")
            }
            self.layer_indeces.push(id as usize);
            Ok(id as usize)
        }
    }

    pub fn add_activation(&mut self, ind: usize, act: ActivationType) -> Result<usize, BuildError> {
        let act_name = match act {
            ActivationType::Linear => "linear",
            ActivationType::Mish => "mish",
            ActivationType::Logistic => "logistic"
        };
        let name = std::ffi::CString::new(act_name).unwrap();
        unsafe {
            let id = pnn_sys::builder_add_activation(
                self.builder,
                ind,
                name.as_ptr()
            );
            if id < 0 {
                return get_error("Could add Activation to TRTBuilder")
            }
            self.layer_indeces.push(id as usize);
            Ok(id as usize)
        }
    }

    pub fn add_shortcut(&mut self, indeces: &Vec<usize>) -> Result<usize, BuildError> {
        unsafe {
            let id = pnn_sys::builder_add_shortcut(
                self.builder,
                indeces.len(),
                indeces.as_ptr()
            );
            if id < 0 {
                return get_error("Could add Shortcut to TRTBuilder")
            }
            self.layer_indeces.push(id as usize);
            Ok(id as usize)
        }
    }

    pub fn add_route(&mut self, indeces: &Vec<usize>) -> Result<usize, BuildError> {
        unsafe {
            let id = pnn_sys::builder_add_route(
                self.builder,
                indeces.len(),
                indeces.as_ptr()
            );
            if id < 0 {
                return get_error("Could add Route to TRTBuilder")
            }
            self.layer_indeces.push(id as usize);
            Ok(id as usize)
        }
    }

    pub fn add_pooling(&mut self, ind: usize,
        stride: usize,
        window_size: usize,
        padding: usize,
        is_max: bool) -> Result<usize, BuildError> {
        unsafe {
            let id = pnn_sys::builder_add_pooling(
                self.builder,
                ind, stride, window_size, padding, is_max as usize
            );
            if id < 0 {
                return get_error("Could add Pooling to TRTBuilder")
            }
            self.layer_indeces.push(id as usize);
            Ok(id as usize)
        }
    }

    pub fn build(&mut self, avg_iters: usize, min_iters: usize, engine_path: String) -> Result<(), BuildError> {
        let name = std::ffi::CString::new(engine_path).unwrap();
        unsafe {
            let res = pnn_sys::builder_build(
                self.builder,
                avg_iters,
                min_iters,
                name.as_ptr()
            );
            if res != 0 {
                return get_error("Couldnt build engine")
            }
        }
        Ok(())
    }


    pub fn dtype(&self) -> cudnnDataType {
        self.data_type.clone()
    }

    pub fn input_size(&self) -> (usize, usize) {
        self.input_size
    }

}

impl Drop for TRTBuilder {
    fn drop(&mut self) {
        unsafe {
            pnn_sys::builder_destroy(self.builder);
        }
    }
}

pub struct TRTEngine {
    // datatype
    data_type: cudnnDataType,
    // Net size, can be usefull for postprocess
    input_size: (usize, usize),
    // Batchsize
    batchsize: usize,
    // Inputs
    inputs: Bindings,
    // Outputs
    outputs: Bindings,
    // Parsers for detections
    det_parsers: HashMap<String, Box<dyn DetectionsParser>>
}

impl TRTEngine {

}

impl Engine for TRTEngine {

    fn forward(&mut self) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn input_bindings(&self) -> Bindings {
        self.inputs.clone()
    }

    fn output_bindings(&self) -> Bindings {
        self.outputs.clone()
    }

    fn add_detections_parser(&mut self, binding_name: &String, parser: Box<dyn DetectionsParser>) {
        self.det_parsers.insert(binding_name.clone(), parser);
    }

    fn batchsize(&self) -> usize {
        self.batchsize
    }

    fn detection_parsers(&self) -> &HashMap<String, Box<dyn DetectionsParser>> {
        &self.det_parsers
    }

    fn dtype(&self) -> cudnnDataType {
        self.data_type.clone()
    }
}