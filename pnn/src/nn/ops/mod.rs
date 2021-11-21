use crate::nn::{
    RuntimeError,
    BuildError,
    Shape,
    LayerShape
};
use std::{
    rc::Rc,
    cell::RefCell
};
use crate::cudnn::{
    Tensor,
    cudnnDataType,
    DevicePtr
};

pub type InputTensor = Rc<RefCell<Tensor>>;
pub type OutputTensor = Rc<RefCell<Tensor>>;


pub trait LayerOp: std::fmt::Debug {
    fn forward(&mut self) -> Result<(), RuntimeError>;
}

pub fn create_otensor(shape: Rc<dyn Shape>, dtype: cudnnDataType) -> Result<OutputTensor, BuildError> {
    let ptr = Rc::new(RefCell::new(
        DevicePtr::new(dtype, shape.size()).map_err(|e| {
            BuildError::Runtime(e)
        })?
    ));

    let tensor_shape: Box<dyn Shape> = Box::new(LayerShape::new(shape.dims()));
    let tensor = Rc::new(RefCell::new(
        Tensor::new(tensor_shape, ptr).map_err(|e| {
            BuildError::Runtime(e)
        })?
    ));
    Ok(tensor)
}

mod conv;
mod batchnorm;
mod activation;
mod pooling;
mod route;
mod shortcut;
mod upsample;
mod convert;

pub use conv::*;
pub use batchnorm::*;
pub use activation::*;
pub use pooling::*;
pub use route::*;
pub use shortcut::*;
pub use upsample::*;
pub use convert::*;