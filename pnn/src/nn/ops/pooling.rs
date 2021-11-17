use crate::cudnn::{Tensor,
    cudnnHandle_t,
    cudnnDataType,
    DevicePtr,
    cudnnError,
    Scale,
};
use crate::nn::{LayerOp, RuntimeError, InputTensor, OutputTensor};
use pnn_sys::{
    cudnnPoolingDescriptor_t,
    cudnnSetPooling2dDescriptor,
    cudnnGetPooling2dForwardOutputDim,
    cudnnDestroyPoolingDescriptor,
    cudnnPoolingForward,
};


use std::{
    rc::Rc,
    cell::RefCell,
    mem::MaybeUninit,
    ptr::addr_of,
    os::raw::{c_void, c_int}
};

#[derive(Debug)]
pub struct PoolingOp {
    input_tensor: InputTensor,
    output_tensor: OutputTensor,
    context: Rc<cudnnHandle_t>,
    desc: cudnnPoolingDescriptor_t,

    scales: Scale
}

const SUPPORTED_DTYPES: [cudnnDataType; 4] = [
    cudnnDataType::HALF,
    cudnnDataType::FLOAT,
    cudnnDataType::DOUBLE,
    cudnnDataType::INT8
];

// TODO: Move bindings to place of use
impl PoolingOp {
    pub fn new(context: Rc<cudnnHandle_t>,
        input_tensor: InputTensor,
        output_tensor: OutputTensor,
        data_type: &cudnnDataType,
        is_max_pool: bool,
        stride_x: usize,
        stride_y: usize,
        padding_x: usize,
        padding_y: usize,
        size_x: usize,
        size_y: usize
    ) -> Result<PoolingOp, RuntimeError> {
        if !SUPPORTED_DTYPES.contains(data_type) {
            return Err(RuntimeError::Other(String::from("Not supported type for pooling")))
        }
        let mut desc: cudnnPoolingDescriptor_t = std::ptr::null_mut() as cudnnPoolingDescriptor_t;
        unsafe {
            let res = pnn_sys::cudnnCreatePoolingDescriptor(&mut desc as *mut cudnnPoolingDescriptor_t);
            if res != 0 {
                return Err(RuntimeError::Cudnn(cudnnError::from(res)));
            }
        };
        let mode = if is_max_pool {pnn_sys::cudnnPoolingMode_t_CUDNN_POOLING_MAX} else {pnn_sys::cudnnPoolingMode_t_CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING};
        unsafe {
            let res = pnn_sys::cudnnSetPooling2dDescriptor(
                desc,
                mode,
                pnn_sys::cudnnNanPropagation_t_CUDNN_NOT_PROPAGATE_NAN,
                size_y as i32, size_x as i32,
                padding_y as i32, padding_x as i32,
                stride_y as i32, stride_x as i32
            );
            if res != 0 {
                return Err(RuntimeError::Cudnn(cudnnError::from(res)));
            }
        }
        unsafe {
            let n: c_int = 0;
            let c: c_int = 0;
            let h: c_int = 0;
            let w: c_int = 0;
            let ret = pnn_sys::cudnnGetPooling2dForwardOutputDim(
                desc,
                input_tensor.borrow().desc(),
                addr_of!(n) as *mut c_int,
                addr_of!(c) as *mut c_int,
                addr_of!(h) as *mut c_int,
                addr_of!(w) as *mut c_int,
            );
            if ret != 0 {
                return Err(RuntimeError::Cudnn(cudnnError::from(ret)));
            }
            let dims: Vec<usize> = vec![n, c, h, w].iter().map(|x| {*x as usize}).collect();
            let target = output_tensor.borrow().shape();
            if dims != target.dims() {
                return Err(RuntimeError::Other(format!("Mismatched shape. CUDNN expect {}x{}x{}x{}, passed {}", n, c, h, w, target)))
            }
        }
        let scales = Scale::new(&data_type, 1., 0.);
        Ok(PoolingOp{input_tensor, output_tensor,
            context, scales, desc})
    }
}

impl LayerOp for PoolingOp {
    
    fn forward(&mut self) -> Result<(), RuntimeError> {
        unsafe {
            let x_desc;
            let x_ptr;
            {   // Allow inplace operations for layer
                x_desc = self.input_tensor.borrow().desc();
                x_ptr = self.input_tensor.borrow_mut().ptr().borrow().ptr();
            }
            let mut y = self.output_tensor.borrow_mut();
            let ret = pnn_sys::cudnnPoolingForward(
                *self.context.as_ref(),
                self.desc,
                self.scales.alpha_ptr(),
                x_desc,
                x_ptr,
                self.scales.beta_ptr(),
                y.desc(),
                y.ptr().borrow().ptr() as *mut c_void
            );
            if ret != 0 {
                return Err(RuntimeError::Cudnn(cudnnError::from(ret)));
            }
        }
        Ok(())
    }
}

impl Drop for PoolingOp {
    fn drop(&mut self) {
        unsafe {
            let res = pnn_sys::cudnnDestroyPoolingDescriptor(self.desc);
            if res != 0 {
                panic!("Couldnt destroy cudnnPooling descriptor")
            }
        }
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
    
        let handle = Rc::new(cudnnCreate().unwrap());
        let mut pool = PoolingOp::new(
            handle.clone(),
            inp.clone(),
            inp.clone(),
            &dtype, true,
            1, 1,
            2, 2,
            5, 5
        ).unwrap();
        pool.forward().unwrap();
    
    }
}