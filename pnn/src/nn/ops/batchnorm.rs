use crate::cudnn::{Tensor,
    cudnnHandle_t,
    cudnnDataType,
    cudnnCreateTensorDescriptor,
    cudnnSetTensor4dDescriptor,
    cudnnDestroyTensorDescriptor,
    cudnnTensorDescriptor_t,
    DevicePtr,
    cudnnError,
    Scale,
};
use crate::nn::{LayerOp, RuntimeError, InputTensor, OutputTensor};

const EPSILON: f64 = 0.00001;

use std::{
    rc::Rc,
    cell::RefCell,
    mem::MaybeUninit,
    ptr::addr_of,
    os::raw::{c_void, c_int}
};

#[derive(Debug)]
pub struct BatchnormOp {
    input_tensor: InputTensor,
    output_tensor: OutputTensor,
    context: Rc<cudnnHandle_t>,
    // bnScaleBiasMeanVarDesc
    scbnvr_desc: cudnnTensorDescriptor_t,
    internal_dtype: cudnnDataType,
    
    scale_ptr: DevicePtr,
    bias_ptr: DevicePtr,
    mean_ptr: DevicePtr,
    var_ptr: DevicePtr,

    scales: Scale

}

type F32Vec = Vec<f32>;

// #TODO: Move bindings to place of use
// #TODO: check dtypes for half inference
impl BatchnormOp {
    pub fn new(context: Rc<cudnnHandle_t>,
        input_tensor: InputTensor,
        output_tensor: OutputTensor,
        data_type: &cudnnDataType,
        channels: usize,
        weights: Option<(&F32Vec, &F32Vec, &F32Vec, &F32Vec)>
    ) -> Result<BatchnormOp, RuntimeError> {
        if input_tensor.borrow().shape().dims() != output_tensor.borrow().shape().dims() {
            return Err(RuntimeError::Other(String::from("Mismatched shape. Batchnorm can work only with same input and output shapes")))
        }

        let internal_dtype = match data_type {
            cudnnDataType::HALF => cudnnDataType::FLOAT,
            x => x.clone()
        };
        let scbnvr_desc = cudnnCreateTensorDescriptor().map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;
        cudnnSetTensor4dDescriptor(scbnvr_desc, 
            internal_dtype.clone(),
            1, channels, 1, 1
        ).map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;
        let mut bias_ptr = DevicePtr::new(internal_dtype.clone(), channels)?;
        let mut scale_ptr = DevicePtr::new(internal_dtype.clone(), channels)?;
        let mut mean_ptr = DevicePtr::new(internal_dtype.clone(), channels)?;
        let mut var_ptr = DevicePtr::new(internal_dtype.clone(), channels)?;
        if let Some(w) = weights {
            bias_ptr.load_with_conversion(&w.0)?;
            scale_ptr.load_with_conversion(&w.1)?;
            mean_ptr.load_with_conversion(&w.2)?;
            var_ptr.load_with_conversion(&w.3)?;
        }

        let scales = Scale::new(&data_type, 1., 0.);
        Ok(BatchnormOp{input_tensor, output_tensor,
            context, scales,
            scale_ptr, bias_ptr,
            mean_ptr, var_ptr,
            internal_dtype, scbnvr_desc})
    }
}

impl LayerOp for BatchnormOp {
    
    fn forward(&mut self) -> Result<(), RuntimeError> {
        unsafe {
            let x_desc;
            let x_ptr;
            {   // Allow inplace operations for layer
                x_desc = self.input_tensor.borrow().desc();
                x_ptr = self.input_tensor.borrow_mut().ptr().borrow().ptr();
            }
            let mut y = self.output_tensor.borrow_mut();

            let ret = pnn_sys::cudnnBatchNormalizationForwardInference(
                *self.context.as_ref(),
                pnn_sys::cudnnBatchNormMode_t_CUDNN_BATCHNORM_SPATIAL,
                self.scales.alpha_ptr(),
                self.scales.beta_ptr(),
                x_desc,
                x_ptr,
                y.desc(),
                y.ptr().borrow().ptr() as *mut c_void,
                self.scbnvr_desc,
                self.scale_ptr.ptr(),
                self.bias_ptr.ptr(),
                self.mean_ptr.ptr(),
                self.var_ptr.ptr(),
                EPSILON
            );
            if ret != 0 {
                return Err(RuntimeError::Cudnn(cudnnError::from(ret)));
            }
        }
        Ok(())
    }
}

impl Drop for BatchnormOp {
    fn drop(&mut self) {
        cudnnDestroyTensorDescriptor(self.scbnvr_desc).expect("Couldnt destroy tensor in batchnorm");
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
        let mut bn = BatchnormOp::new(
            handle,
            inp.clone(),
            inp.clone(),
            &dtype,
            32,
            None
        ).unwrap();
        bn.forward().unwrap();
    
    }
}