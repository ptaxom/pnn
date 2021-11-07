use crate::nn::Shape;
use std::{os::raw::c_void};
use pnn_sys::{cudnnTensorDescriptor_t};
use crate::cudnn::*;

//Container to store device ptr and tensor information in conjunction 
pub struct Tensor {
    pub shape: Box<dyn Shape>,
    device_data_ptr: *mut c_void,
    tensor_desc: cudnnTensorDescriptor_t
}

impl Tensor {
    pub fn new(shape: Box<dyn Shape>) -> Result<Self, Box<dyn std::error::Error>> {
        // allocate CUDA memory at GPU
        // Currently support only for f32
        let device_data_ptr = cudaMalloc(std::mem::size_of::<f32>() * shape.size())?;
        let tensor_desc = cudnnCreateTensorDescriptor()?;
        cudnnSetTensor4dDescriptor(
            tensor_desc,
            cudnnDataType::FLOAT,
            shape.N(),
            shape.C(),
            shape.H().unwrap_or(1),
            shape.W().unwrap_or(1)
        )?;
        Ok(Self{shape, device_data_ptr, tensor_desc})
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        cudaFree(self.device_data_ptr).expect("Couldnt free allocated CUDA memory");
        cudnnDestroyTensorDescriptor(self.tensor_desc).expect("Could free tensor descriptor");
    }
}