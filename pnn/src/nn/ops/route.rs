use crate::cudnn::{Tensor,
    cudnnHandle_t,
    cudnnDataType,
    DevicePtr,
    cudaError,
    cudaStream_t,
    cudaMemcpyAsync,
    cudaMemcpyKind,
    cudnnGetStream,
    cudnnSizeOf
};
use crate::nn::{LayerOp, RuntimeError, InputTensor, OutputTensor};


use std::{
    rc::Rc,
    cell::RefCell,
    os::raw::{c_void}
};

pub struct RouteOp {
    input_tensors: Vec<InputTensor>,
    output_tensor: OutputTensor,
    b: usize,
    channel_size: usize,
    stream: cudaStream_t

}


// TODO: Move bindings to place of use
impl RouteOp {
    pub fn new(context: Rc<cudnnHandle_t>,
        input_tensors: Vec<InputTensor>,
        output_tensor: OutputTensor,
    ) -> Result<RouteOp, RuntimeError> {
        let stream = cudnnGetStream(*context.as_ref()).map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;
        let shape = output_tensor.borrow().shape();
        let b = shape.N();
        let channel_size = cudnnSizeOf(&output_tensor.borrow().data_type()) * shape.H().unwrap() * shape.W().unwrap();

        Ok(RouteOp{input_tensors, output_tensor,
            stream, channel_size, b})
    }
}

impl LayerOp for RouteOp {
    
    fn forward(&mut self) -> Result<(), RuntimeError> {
        let mut dst_ptr = self.output_tensor.borrow_mut().ptr().borrow_mut().ptr() as *mut c_void;
        for _ in 0..self.b {
            for tensor_id in 0..self.input_tensors.len() {
                let tensor = &self.input_tensors[tensor_id];
                let ptr = tensor.borrow_mut().ptr();
                let n_channels = tensor.borrow().shape().C();
                let mut src_ptr = ptr.borrow().ptr();
                for _ in 0..n_channels {
                    cudaMemcpyAsync(dst_ptr,
                        src_ptr,
                        self.channel_size,
                        cudaMemcpyKind::DeviceToDevice,
                        self.stream).map_err(|e| {
                            RuntimeError::Cuda(e)
                        })?;
                    unsafe {
                        dst_ptr = dst_ptr.offset(self.channel_size as isize);
                        src_ptr = src_ptr.offset(self.channel_size as isize);
                    }
                }
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
        const dtype: cudnnDataType = cudnnDataType::FLOAT;

        
        let x_data = Rc::new(RefCell::new(DevicePtr::new(dtype.clone(), 4 * 512 * 16 * 16).unwrap()));
        let inp = Rc::new(RefCell::new(Tensor::new(Box::new(LayerShape::from_nchw(4, 512, 16, 16)), x_data.clone()).unwrap()));

        let y_data = Rc::new(RefCell::new(DevicePtr::new(dtype.clone(), 4 * 2048 * 16 * 16).unwrap()));
        let outp = Rc::new(RefCell::new(Tensor::new(Box::new(LayerShape::from_nchw(4, 2048, 16, 16)), y_data.clone()).unwrap()));
    
        let handle = Rc::new(cudnnCreate().unwrap());
        let mut route = RouteOp::new(
           handle,
            vec![inp.clone(),inp.clone(), inp.clone(), inp.clone()],
            outp
        ).unwrap();
        route.forward().unwrap();
    }
}