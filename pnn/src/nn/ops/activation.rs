use crate::cudnn::{Tensor,
    cudnnHandle_t,
    cudnnDataType,
    DevicePtr,
    cudnnError,
    cudaError,
    Scale,
    cudnnGetStream,
    cudaStream_t
};
use crate::nn::{LayerOp, RuntimeError, InputTensor, OutputTensor, ActivationType};

const EPSILON: f64 = 0.00001;

use std::{
    rc::Rc,
    cell::RefCell,
    os::raw::{c_void, c_int}
};
use pnn_sys::{cudaError_t, cudnnActivationDescriptor_t};
type ExternCudaCall = fn(*mut c_void, usize, cudaStream_t) -> cudaError_t;

pub struct ActivationOp {
    tensor: OutputTensor,
    context: Rc<cudnnHandle_t>,
    data_type: cudnnDataType,

    activation: ActivationType,
    scales: Scale,
    
    cuda_func: Option<ExternCudaCall>,
    cudnn_act_desc: Option<cudnnActivationDescriptor_t>,
    stream: Option<cudaStream_t>

}

fn safe_mish_fp16(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t {
    unsafe {
        return pnn_sys::activation_mish_fp16(data, elements, stream);
    }
}

fn safe_mish_fp32(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t {
    unsafe {
        return pnn_sys::activation_mish_fp32(data, elements, stream);
    }
}

fn safe_mish_fp64(data: *mut c_void, elements: usize, stream: cudaStream_t) -> cudaError_t {
    unsafe {
        return pnn_sys::activation_mish_fp64(data, elements, stream);
    }
}

// TODO: Move bindings to place of use
impl ActivationOp {
    pub fn new(context: Rc<cudnnHandle_t>,
        tensor: OutputTensor,
        data_type: &cudnnDataType,
        activation: &ActivationType
    ) -> Result<ActivationOp, RuntimeError> {
        let data_type = data_type.clone();
        let activation = activation.clone();

        let scales = Scale::new(&data_type, 1., 0.);

        let mut cuda_func = None;
        let mut stream = None;
        let mut cudnn_act_desc = None;

        if activation == ActivationType::Mish {
            println!("MISH");
            cuda_func = Some(match data_type {
                cudnnDataType::HALF => safe_mish_fp16,
                cudnnDataType::FLOAT => safe_mish_fp32,
                cudnnDataType::DOUBLE => safe_mish_fp64,
                _ => return Err(RuntimeError::Other(String::from("Unsupported data type for Mish activation")))
            });
            stream = Some(cudnnGetStream(*context.as_ref()).unwrap());
        } else if activation == ActivationType::Logistic {
            println!("Logistic");
            unsafe {
                let mut act_desc: cudnnActivationDescriptor_t = std::ptr::null_mut() as cudnnActivationDescriptor_t;
                let res = pnn_sys::cudnnCreateActivationDescriptor(&mut act_desc as *mut cudnnActivationDescriptor_t);
                if res != 0 {
                    return Err(RuntimeError::Cudnn(cudnnError::from(res)));
                }
                cudnn_act_desc = Some(act_desc);

                let res = pnn_sys::cudnnSetActivationDescriptor(
                    act_desc,
                    pnn_sys::cudnnActivationMode_t_CUDNN_ACTIVATION_SIGMOID,
                    pnn_sys::cudnnNanPropagation_t_CUDNN_NOT_PROPAGATE_NAN,
                    0.
                );
                if res != 0 {
                    return Err(RuntimeError::Cudnn(cudnnError::from(res)));
                }
            }
        }
        
        let cond = cuda_func != None && cudnn_act_desc != None;
        assert!(!cond, "Couldnt use cuda and cudnn in same activation");
        Ok(ActivationOp{context, tensor, data_type, activation, scales, cuda_func, cudnn_act_desc, stream})
    }
}

impl LayerOp for ActivationOp {
    
    fn forward(&mut self) -> Result<(), RuntimeError> {
        if let Some(func) = self.cuda_func {
            let tensor = self.tensor.borrow_mut().ptr();
            let array = tensor.borrow();
            let err = func(array.ptr() as *mut c_void, array.capacity(), self.stream.unwrap());
            if err != 0 {
                return Err(RuntimeError::Cuda(cudaError::from(err)));
            }
        }
        if let Some(desc) = self.cudnn_act_desc {
            unsafe {
                let mut x = self.tensor.borrow_mut();
                let x_desc = x.desc();
                let x_data = x.ptr().borrow().ptr();

                let err = pnn_sys::cudnnActivationForward(
                    *self.context.as_ref(),
                    desc,
                    self.scales.alpha_ptr(),
                    x_desc,
                    x_data,
                    self.scales.beta_ptr(),
                    x_desc,
                    x_data as *mut c_void
                    
                );
                if err != 0 {
                    return Err(RuntimeError::Cudnn(cudnnError::from(err)));
                }
            }
        }
        Ok(())
    }
}

impl Drop for ActivationOp {
    fn drop(&mut self) {
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_act(activation: ActivationType) {
        use crate::cudnn::*;
        use crate::nn::LayerShape;

        let dtype = cudnnDataType::FLOAT;
        let x_data = Rc::new(RefCell::new(DevicePtr::new(dtype.clone(), 4 * 64 * 416 * 416).unwrap()));
        let inp = Rc::new(RefCell::new(Tensor::new(Box::new(LayerShape::from_nchw(4, 32, 416, 416)), x_data.clone()).unwrap()));
        let handle = Rc::new(cudnnCreate().unwrap());
    
        let mut act = ActivationOp::new(handle, inp, &dtype, &activation).unwrap();
        act.forward().unwrap();
    }

    #[test]
    fn test_lin() {
        test_act(ActivationType::Linear);
    }

    #[test]
    fn test_mish() {
        test_act(ActivationType::Mish);
    }

    #[test]
    fn test_logistic() {
        test_act(ActivationType::Logistic);
    }
}