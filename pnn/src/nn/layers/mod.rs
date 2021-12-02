use std::{
    collections::HashMap,
    self,
    any::Any,
    rc::Rc,
    cell::RefCell
};

use crate::nn::shape::*;
use crate::nn::errors::*;
use crate::nn::{Engine, CUDNNEngine, BuildInformation, TRTBuilder};
use crate::nn::ops::{LayerOp, OutputTensor};
use crate::parsers::DeserializationError;
use crate::cudnn::{cudnnHandle_t, cudnnDataType};

pub trait Layer {
    fn name(&self) -> String;

    fn shape(&self) -> Option<Rc<dyn Shape>>;

    fn propose_name() -> String where Self: Sized;

    fn from_config(config: HashMap<String, String>) -> Result<Box<dyn Layer>, DeserializationError> where Self: Sized;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn infer_shape(&mut self, input_shapes: Vec<Rc<dyn Shape>>) -> Result<(), ShapeError>;

    fn ltype(&self) -> LayerType;

    fn input_indices(&self, position: usize) -> Result<Vec<usize>, BuildError> {
        if position == 0 {
            return Err(BuildError::Deserialization(DeserializationError(String::from("Couldnt compute input index for first layer"))))
        }
        Ok(vec![position - 1])
    }

    fn build_cudnn(&mut self, 
        engine: Rc<RefCell<CUDNNEngine>>,
        indeces: Vec<usize>,
        has_depend_layers: bool
    ) -> Result<(), BuildError>;

    fn build_trt(&mut self, 
        engine: Rc<RefCell<TRTBuilder>>,
        indeces: Vec<usize>
    ) -> Result<(), BuildError>;

    // Initialize weights using darknet model file. Consume initial offset and return new
    fn load_darknet_weights(&mut self, offset: usize, _bytes: &Vec<u8>) -> Result<usize, BuildError> {
        Ok(offset)
    }
}

#[derive(Debug, PartialEq)]
pub enum LayerType {
    Input,
    Convolutional,
    Maxpool,
    Route,
    Yolo,
    Shortcut,
    Upsample,
    Unknown
}

impl From<&String> for LayerType {
    fn from(layer_type: &String) -> Self {

        if layer_type == "input" || layer_type == "net" {
            return LayerType::Input
        } else if layer_type == "convolutional" {
            return LayerType::Convolutional
        } else if layer_type == "maxpool" {
            return LayerType::Maxpool
        } else if layer_type == "route" {
            return LayerType::Route
        } else if layer_type == "upsample" {
            return LayerType::Upsample
        } else if layer_type == "yolo" {
            return LayerType::Yolo
        } else if layer_type == "shortcut" {
            return LayerType::Shortcut
        }
        LayerType::Unknown
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum ActivationType {
    Linear,
    Mish,
    Logistic
}


impl std::convert::TryFrom<&String> for ActivationType {
    type Error = DeserializationError;
    fn try_from(layer_type: &String) -> Result<Self, Self::Error> {

        if layer_type == "linear" {
            return Ok(ActivationType::Linear)
        } else if layer_type == "mish" {
            return Ok(ActivationType::Mish)
        } else if layer_type == "logistic" {
            return Ok(ActivationType::Logistic)
        }

        Err(DeserializationError(format!("Couldnt parse activation from {}", layer_type)))
    }
}


mod input;
mod convolutional;
mod shortcut;
mod route;
mod maxpool;
mod upsample;
mod yolo;

pub use input::*;
pub use convolutional::*;
pub use shortcut::*;
pub use route::*;
pub use maxpool::*;
pub use upsample::*;
pub use yolo::*;