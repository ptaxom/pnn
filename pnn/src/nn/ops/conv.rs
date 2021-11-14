use crate::cudnn::{Tensor,
    cudnnHandle_t,
    cudnnFilterDescriptor_t,
    cudnnCreateFilterDescriptor,
    cudnnDestroyFilterDescriptor,
    cudnnSetFilter4dDescriptor,
    cudnnDataType,
    cudnnConvolutionDescriptor_t,
    cudnnCreateConvolutionDescriptor,
    cudnnDestroyConvolutionDescriptor,
    cudnnSetConvolution2dDescriptor,
    DevicePtr
};
use crate::nn::{LayerOp, RuntimeError, InputTensor, OutputTensor};

use std::{
    rc::Rc,
    cell::RefCell
};

pub struct ConvolutionOp {
    input_tensor: InputTensor,
    output_tensor: OutputTensor,
    context: Rc<cudnnHandle_t>,

    filter_desc: cudnnFilterDescriptor_t,
    filter_data: DevicePtr,

    conv_desc: cudnnConvolutionDescriptor_t,

}

// TODO: Move bindings to place of use
impl ConvolutionOp {
    pub fn new(input_tensor: InputTensor,
        output_tensor: OutputTensor,
        context: Rc<cudnnHandle_t>,
        data_type: &cudnnDataType,
        filters: usize,
        input_channels: usize,
        size_y: usize,
        size_x: usize,
        pad_y: usize,
        pad_x: usize,
        stride_y: usize,
        stride_x: usize
    ) -> Result<ConvolutionOp, RuntimeError> {
        let filter_desc = cudnnCreateFilterDescriptor().map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;
        cudnnSetFilter4dDescriptor(filter_desc, data_type.clone(), filters, input_channels, size_y, size_x).map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;

        let filter_data = DevicePtr::new(data_type.clone(), filters * size_x * size_y)?;
        let conv_desc = cudnnCreateConvolutionDescriptor().map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;
        cudnnSetConvolution2dDescriptor(conv_desc, pad_y, pad_x, stride_y, stride_x, 1, 1, data_type.clone()).map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;


        Ok(ConvolutionOp{input_tensor, output_tensor, context, filter_desc, filter_data, conv_desc})
    }
}

impl LayerOp for ConvolutionOp {
    
    fn forward(&mut self) -> Result<(), RuntimeError> {
        Ok(())
    }

}