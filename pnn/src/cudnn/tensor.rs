use crate::nn::shape::Shape;
use std::{os::raw::c_void};

pub struct Tensor {
    shape: Box<dyn Shape>,
    data_ptr: *mut c_void,
}

impl Tensor {

}