use crate::nn::RuntimeError;
use std::{
    rc::Rc,
    cell::RefCell
};
use crate::cudnn::Tensor;

pub type InputTensor = Rc<RefCell<Tensor>>;
pub type OutputTensor = Rc<RefCell<Tensor>>;

pub trait LayerOp {
    fn forward(&mut self) -> Result<(), RuntimeError>;
}

mod conv;
mod batchnorm;

pub use conv::*;
pub use batchnorm::*;