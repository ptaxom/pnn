use crate::nn::RuntimeError;

pub trait LayerOp {
    fn forward(&mut self) -> Result<(), RuntimeError>;
}