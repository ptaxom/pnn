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

    pub fn finilize_layer(&mut self, id: usize) {
        self.layer_indeces.push(id);
    }

    pub fn last_op_id(&self, layer_id: usize) -> usize {
        self.layer_indeces[layer_id]
    }

    pub fn add_input(&mut self, name: &String, shape: Rc<dyn Shape>) -> Result<(), BuildError> {
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
        }
        Ok(())
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
        padding: usize,
        stride: usize,
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
                padding,
                stride,
                kernels.as_ptr(),
                biases.as_ptr()
            );
            if id < 0 {
                return get_error("Could add Convolution to TRTBuilder")
            }
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
            Ok(id as usize)
        }
    }

    pub fn add_pooling(&mut self, ind: usize,
        stride: usize,
        window_size: usize,
        padding: usize,
        is_max: bool
    ) -> Result<usize, BuildError> {       
        unsafe {
            let id = pnn_sys::builder_add_pooling(
                self.builder,
                ind, stride, window_size, padding / 2, is_max as usize
            );
            if id < 0 {
                return get_error("Could add Pooling to TRTBuilder")
            }
            Ok(id as usize)
        }
    }

    pub fn add_upsample(&mut self, ind: usize, stride: usize) -> Result<usize, BuildError> {
        unsafe {
            let id = pnn_sys::builder_add_upsample(self.builder, ind, stride);
            if id < 0 {
                return get_error("Could add Upsample to TRTBuilder")
            }
            Ok(id as usize)
        }
    }

    pub fn build(&mut self, avg_iters: usize, min_iters: usize, engine_path: &String) -> Result<(), BuildError> {
        let name = std::ffi::CString::new(engine_path.clone()).unwrap();
        unsafe {
            let res = pnn_sys::builder_build(
                self.builder,
                avg_iters,
                min_iters,
                name.as_ptr()
            );
            if res == 0 {
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
    // Batchsize
    batchsize: usize,
    // Inputs
    inputs: Bindings,
    // Outputs
    outputs: Bindings,
    // Parsers for detections
    det_parsers: HashMap<String, Box<dyn DetectionsParser>>,
    // Engine itselft
    engine: *mut c_void,
    // Cuda stream for inference
    stream: crate::cudnn::cudaStream_t,
    // Shapes for bindings, // TODO
}

impl TRTEngine {
    pub fn new(engine_path: &String) -> Result<TRTEngine, RuntimeError> {
        let stream = crate::cudnn::cudaStreamCreate().map_err(|e| {RuntimeError::Cuda(e)})?;
        let mut inputs = HashMap::new();
        let mut outputs = HashMap::new();
        let det_parsers = HashMap::new();
        
        let mut engine = std::ptr::null_mut();
        unsafe {
            let cpath = std::ffi::CString::new(engine_path.clone()).unwrap();
            engine = pnn_sys::engine_create(cpath.as_ptr(), stream);
        }
        if engine == std::ptr::null_mut() {
            return Err(RuntimeError::Other(String::from("Couldnt create TRTEngine")))
        }
        let batchsize = unsafe { pnn_sys::engine_batchsize(engine)};

        let n_bindings = unsafe { pnn_sys::engine_n_bindings(engine)};

        for i in 0..n_bindings {
            let info: pnn_sys::BindingInfo;
            unsafe {
                info = pnn_sys::engine_get_info(engine, i);
            }
            use std::ffi::{CStr, CString};
            let c_name: &CStr = unsafe {CStr::from_ptr(info.name)};
            let name = c_name.to_str().unwrap().to_owned();

            let size = info.batchsize * info.channels * info.height * info.width;
            let dev_ptr = DevicePtr::new(cudnnDataType::FLOAT, size)?;
            unsafe {
                pnn_sys::engine_add_ptr(engine, dev_ptr.ptr());
            }

            let dev_memory = Rc::new(RefCell::new(dev_ptr));
            if info.is_input != 0 {
                inputs.insert(name, dev_memory);
            } else {
                outputs.insert(name, dev_memory);
            }
        }

        Ok(TRTEngine{stream, inputs, outputs, det_parsers, engine, batchsize})
    } 
}

impl Drop for TRTEngine {
    fn drop(&mut self) {
        unsafe {
            pnn_sys::engine_destroy(self.engine);
        }
        crate::cudnn::cudaStreamDestroy(self.stream).unwrap();
    }
}

impl Engine for TRTEngine {

    fn forward(&mut self) -> Result<(), RuntimeError> {
        unsafe {
            pnn_sys::engine_forward(self.engine)
        }
        Ok(())
    }

    fn inputs(&self) -> Vec<String> {
        self.inputs.keys().cloned().collect()
    }

    fn input_binding(&self, name: &String) -> Option<Rc<RefCell<DevicePtr>>> {
        self.inputs.get(name).map(|x| {x.clone()})
    }

    fn output_binding(&self, name: &String) -> Option<Rc<RefCell<DevicePtr>>> {
        self.outputs.get(name).map(|x| {x.clone()})
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
}