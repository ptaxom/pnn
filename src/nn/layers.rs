use std::{
    collections::HashMap,
    self,
    error::Error,
    fmt,
    any::Any
};

use crate::nn::shape::*;

#[derive(Debug)]
pub struct DeserializationError {
    description: String,
}

impl fmt::Display for DeserializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl Error for DeserializationError {}

pub trait Layer {
    fn name(&self) -> String;

    fn shape(&self) -> Option<Box<dyn Shape>>;

    fn propose_name() -> String where Self: Sized;

    fn from_config(config: HashMap<String, String>) -> Result<Box<dyn Layer>, DeserializationError> where Self: Sized;

    fn as_any(&self) -> &dyn Any where Self: Sized;

    fn infer_shape(&mut self, input_shapes: Vec<&dyn Shape>);
}

//Input layer for most NNs
#[derive(Debug)]
pub struct InputLayer {
    // Layer name
    name: String,
    // Layer shape
    shape: Option<Box<dyn Shape>>
}