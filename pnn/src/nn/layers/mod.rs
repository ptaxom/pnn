use std::{
    collections::HashMap,
    self,
    any::Any,
    rc::Rc
};

use crate::nn::shape::*;
use crate::parsers::DeserializationError;

pub trait Layer {
    fn name(&self) -> String;

    fn shape(&self) -> Option<Rc<dyn Shape>>;

    fn propose_name() -> String where Self: Sized;

    fn from_config(config: HashMap<String, String>) -> Result<Box<dyn Layer>, DeserializationError> where Self: Sized;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn infer_shape(&mut self, input_shapes: Vec<&dyn Shape>);
}


mod input;
pub use input::*;