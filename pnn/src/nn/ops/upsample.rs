use crate::cudnn::{Tensor,
    cudnnHandle_t,
    cudnnDataType,
    cudaStream_t,
    cudnnGetStream,
    DevicePtr,
    cudaError
};
use crate::nn::{LayerOp, RuntimeError, InputTensor, OutputTensor};


use std::{
    rc::Rc,
    cell::RefCell,
    mem::MaybeUninit,
    ptr::addr_of,
    os::raw::{c_void, c_int}
};


use pnn_sys::cudaError_t;

type ExternCudaCall = fn(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t;

fn safe_upsample_fp16(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t {
    unsafe {
        return pnn_sys::upsample_forward_fp16(input, n, c, h, w, stride, scale, output, stream);
    }
}

fn safe_upsample_fp32(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t {
    unsafe {
        return pnn_sys::upsample_forward_fp32(input, n, c, h, w, stride, scale, output, stream);
    }
}

fn safe_upsample_fp64(input: *mut c_void, n: usize, c: usize, h: usize, w: usize, stride: usize, scale: f32, output: *mut c_void, stream: cudaStream_t) -> cudaError_t {
    unsafe {
        return pnn_sys::upsample_forward_fp64(input, n, c, h, w, stride, scale, output, stream);
    }
}

pub struct UpsampleOp {
    input_tensor: InputTensor,
    output_tensor: OutputTensor,
    stride: usize, 
    stream: cudaStream_t,
    scale: f32,
    kernel: ExternCudaCall,
    nchw: (usize, usize, usize, usize)
}

const SUPPORTED_DTYPES: [cudnnDataType; 3] = [
    cudnnDataType::HALF,
    cudnnDataType::FLOAT,
    cudnnDataType::DOUBLE
];

impl UpsampleOp {
    pub fn new(context: Rc<cudnnHandle_t>,
        input_tensor: InputTensor,
        output_tensor: OutputTensor,
        stride: usize,
        scale: f32
    ) -> Result<UpsampleOp, RuntimeError> {
        if !SUPPORTED_DTYPES.contains(&output_tensor.borrow().data_type()) {
            return Err(RuntimeError::Other(String::from("Not supported type for pooling")))
        }
        let stream = cudnnGetStream(*context.as_ref()).map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;
        let kernel = match output_tensor.borrow().data_type() {
            cudnnDataType::HALF => safe_upsample_fp16,
            cudnnDataType::FLOAT => safe_upsample_fp32,
            _ => safe_upsample_fp64,
        };
        let shape = input_tensor.borrow().shape();
        let nchw = (
            shape.N(), shape.C(),
            shape.H().unwrap(), shape.W().unwrap()
        );
  
        Ok(UpsampleOp{input_tensor, output_tensor, stride, scale, stream, kernel, nchw})
    }
}

impl LayerOp for UpsampleOp {
    
    fn forward(&mut self) -> Result<(), RuntimeError> {
        let kernel = self.kernel;
        let res = kernel(
            self.input_tensor.borrow_mut().ptr().borrow().ptr() as *mut c_void,
            self.nchw.0, self.nchw.1, self.nchw.2, self.nchw.3,
            self.stride, self.scale,
            self.output_tensor.borrow_mut().ptr().borrow().ptr() as *mut c_void,
            self.stream
        );
        if res != 0 {
            return Err(RuntimeError::Cuda(cudaError::from(res)));
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
        let x_data = Rc::new(RefCell::new(DevicePtr::new(dtype.clone(), 4 * 512 * 16 * 16).unwrap()));
        let inp = Rc::new(RefCell::new(Tensor::new(Box::new(LayerShape::from_nchw(4, 512, 16, 16)), x_data.clone()).unwrap()));

        let y_data = Rc::new(RefCell::new(DevicePtr::new(dtype.clone(), 4 * 512 * 32 * 32).unwrap()));
        let outp = Rc::new(RefCell::new(Tensor::new(Box::new(LayerShape::from_nchw(4, 512, 32, 32)), y_data.clone()).unwrap()));
    
        let handle = Rc::new(cudnnCreate().unwrap());
        let mut upsample = UpsampleOp::new(
            handle.clone(),
            inp,
            outp, 2, 1.
        ).unwrap();
        upsample.forward().unwrap();
    }
}