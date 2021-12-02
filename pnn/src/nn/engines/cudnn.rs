use crate::cudnn::{cudnnCreate, cudnnHandle_t, cudnnDataType};
use crate::nn::{DetectionsParser, BoundingBox, RuntimeError, BuildError};
use crate::cudnn::{DevicePtr};
use crate::nn::ops::{LayerOp, OutputTensor};
use crate::nn::Engine;

use std::{
    rc::Rc,
    cell::RefCell,
    collections::HashMap
};

#[derive(Debug, Clone)]
pub struct BuildInformation {
    // Output tensor
    pub tensor: OutputTensor,
    // Can be used for next layers both as input and output
    pub reusable: bool
}

type Bindings = HashMap<String, DevMemory>;
type DevMemory = Rc<RefCell<DevicePtr>>;

pub struct CUDNNEngine {
    // cudnn context for operations execution
    context: Rc<cudnnHandle_t>,
    // datatype
    data_type: cudnnDataType,
    // Net size, can be usefull for postprocess
    input_size: (usize, usize),
    // Batchsize
    batchsize: usize,
    // Inference operations
    ops: Vec<Box<dyn LayerOp>>,
    // Tensors, needed for linking
    information: Vec<BuildInformation>,
    // Inputs
    inputs: Bindings,
    // Outputs
    outputs: Bindings,
    // Parsers for detections
    det_parsers: HashMap<String, Box<dyn DetectionsParser>>
}

impl CUDNNEngine {

    pub fn new(data_type: cudnnDataType, batchsize: usize, input_size: (usize, usize)) -> Result<CUDNNEngine, BuildError> {
        let context = Rc::new(cudnnCreate().map_err(|e| {
            BuildError::Runtime(RuntimeError::Cudnn(e))
        })?);
        let ops = Vec::new();
        let information = Vec::new();
        let inputs = HashMap::new();
        let outputs = HashMap::new();
        let det_parsers = HashMap::new();

        Ok(CUDNNEngine{context, batchsize, input_size, data_type, ops, information, inputs, outputs, det_parsers})
    }

    pub fn get_info(&self, index: usize) -> BuildInformation {
        self.information[index].clone()
    }

    pub fn add_layer(&mut self, ops: Vec<Box<dyn LayerOp>>, info: BuildInformation) {
        let mut lops = ops;
        self.ops.append(&mut lops);
        self.information.push(info);
    }

    pub fn add_input(&mut self, name: &String, memory: DevMemory) {
        self.inputs.insert(name.clone(), memory);
    }

    pub fn add_output(&mut self, name: &String, memory: DevMemory) {
        self.outputs.insert(name.clone(), memory);
    }

    pub fn dtype(&self) -> cudnnDataType {
        self.data_type.clone()
    }

    pub fn context(&self) -> Rc<cudnnHandle_t> {
        self.context.clone()
    }

    pub fn input_size(&self) -> (usize, usize) {
        self.input_size
    }

}

impl Engine for CUDNNEngine {

    fn forward(&mut self) -> Result<(), RuntimeError> {
        for op in &mut self.ops {
            op.forward()?
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