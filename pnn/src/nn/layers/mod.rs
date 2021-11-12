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

    fn infer_shape(&mut self, input_shapes: Vec<Rc<dyn Shape>>) -> Result<(), ShapeError>;

    fn layer_type(&self) -> LayerType;
}

#[derive(Debug, PartialEq)]
pub enum LayerType {
    Input,
    Convolutional,
    Maxpool,
    Route,
    YoloLayer,
    Shortcut,
    Upsample,
    Unknown
}

impl From<&String> for LayerType {
    fn from(layer_type: &String) -> Self {
        let mut map: HashMap<String, LayerType> = HashMap::new();
        map.insert(String::from("input"), LayerType::Input);
        map.insert(String::from("net"), LayerType::Input);
        map.insert(String::from("convolutional"), LayerType::Convolutional);
        map.insert(String::from("maxpool"), LayerType::Maxpool);
        map.insert(String::from("route"), LayerType::Route);
        map.insert(String::from("shortcut"), LayerType::Shortcut);
        map.insert(String::from("upsample"), LayerType::Upsample);
        map.insert(String::from("yolo"), LayerType::YoloLayer);

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
            return LayerType::YoloLayer
        }
        LayerType::Unknown
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