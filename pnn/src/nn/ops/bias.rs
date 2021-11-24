use crate::cudnn::{Tensor,
    cudnnHandle_t,
    cudnnDataType,
    DevicePtr,
    cudaError,
    cudaStream_t,
    cudnnGetStream
};
use crate::nn::{LayerOp, RuntimeError, InputTensor, OutputTensor};


use std::{
    rc::Rc,
    cell::RefCell,
    os::raw::{c_void, c_int}
};

#[derive(Debug)]
pub struct BiasOp {
    input_tensor: InputTensor,
    output_tensor: OutputTensor,
    stream: cudaStream_t,
    dtype: cudnnDataType,
    channel_size: usize,
    channels: usize,
    bias_ptr: DevicePtr,

}

type F32Vec = Vec<f32>;

// TODO: Move bindings to place of use
impl BiasOp {
    pub fn new(context: Rc<cudnnHandle_t>,
        input_tensor: InputTensor,
        output_tensor: OutputTensor,
        data_type: &cudnnDataType,
        weights: &F32Vec
    ) -> Result<BiasOp, RuntimeError> {
        if input_tensor.borrow().shape().dims() != output_tensor.borrow().shape().dims() {
            return Err(RuntimeError::Other(String::from("Mismatched shape. Bias can work only with same input and output shapes")))
        }
        let shape = input_tensor.borrow().shape();
        let channels = shape.C();
        let channel_size = shape.H().unwrap() * shape.W().unwrap();
        let stream = cudnnGetStream(*context.as_ref()).map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;

        let mut bias_ptr = DevicePtr::new(data_type.clone(), channels)?;
        bias_ptr.load_with_conversion(weights)?;

        let dtype = data_type.clone();

        Ok(BiasOp{input_tensor, output_tensor,
            stream, bias_ptr, dtype, channels, channel_size})
    }
}

impl LayerOp for BiasOp {
    fn forward(&mut self) -> Result<(), RuntimeError> {
        unsafe {
            let inp_ptr = self.input_tensor.borrow_mut().ptr();
            let ret = pnn_sys::add_bias(
                inp_ptr.borrow().ptr() as *mut c_void,
                self.bias_ptr.ptr() as *mut c_void,
                inp_ptr.borrow().capacity(),
                self.channels,
                self.channel_size,
                self.dtype.clone() as usize,
                self.stream
            );
            if ret != 0 {
                return Err(RuntimeError::Cuda(cudaError::from(ret)));
            }
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_test() {
        use crate::cudnn::*;
        use crate::nn::LayerShape;

        let dtype = cudnnDataType::FLOAT;
        let x_data = Rc::new(RefCell::new(DevicePtr::new(dtype.clone(), 4 * 32 * 416 * 416).unwrap()));
        let inp = Rc::new(RefCell::new(Tensor::new(Box::new(LayerShape::from_nchw(4, 32, 416, 416)), x_data.clone()).unwrap()));
    
        let handle = Rc::new(cudnnCreate().unwrap());
        let mut bn = BiasOp::new(
            handle,
            inp.clone(),
            inp.clone(),
            &dtype,
            None
        ).unwrap();
        bn.forward().unwrap();
    
    }
}