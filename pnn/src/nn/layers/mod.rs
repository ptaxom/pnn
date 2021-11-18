use std::{
    collections::HashMap,
    self,
    any::Any,
    rc::Rc,
    cell::RefCell
};

use crate::nn::shape::*;
use crate::nn::errors::*;
use crate::nn::ops::{LayerOp, OutputTensor};
use crate::parsers::DeserializationError;
use crate::cudnn::{cudnnHandle_t, cudnnDataType};

#[derive(Debug)]
pub struct BuildInformation {
    // Output tensor
    tensor: OutputTensor,
    // Can be used for next layers both as input and output
    reusable: bool
}

pub trait Layer {
    fn name(&self) -> String;

    fn shape(&self) -> Option<Rc<dyn Shape>>;

    fn propose_name() -> String where Self: Sized;

    fn from_config(config: HashMap<String, String>) -> Result<Box<dyn Layer>, DeserializationError> where Self: Sized;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn infer_shape(&mut self, input_shapes: Vec<Rc<dyn Shape>>) -> Result<(), ShapeError>;

    fn layer_type(&self) -> LayerType;

    fn input_indices(&self, position: usize) -> Result<Vec<usize>, DeserializationError> {
        if position == 0 {
            return Err(DeserializationError(String::from("Couldnt compute input index for first layer")))
        }
        Ok(vec![position - 1])
    }

    fn get_build_information(&self) -> BuildInformation;

    fn get_operations(&mut self) -> &mut Vec<Box<dyn LayerOp>>;

    fn forward(&mut self) -> Result<(), RuntimeError> {
        for op in self.get_operations() {
            op.forward()?;
        }
        Ok(())
    }

    fn build(&mut self, 
        context: Rc<cudnnHandle_t>,
        data_type: cudnnDataType,
        info: Vec<BuildInformation>,
        has_depend_layers: bool
    ) -> Result<(), BuildError>;

    // Initialize weights using darknet model file. Consume initial offset and return new
    fn load_darknet_weights(&mut self, offset: usize, bytes: &Vec<u8>) -> Result<usize, BuildError> {
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