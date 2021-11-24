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
    cudnnConvolutionFwdAlgo_t,
    DevicePtr,
    cudnnError,
    Scale,
};
use crate::nn::{LayerOp, RuntimeError, InputTensor, OutputTensor};

use std::{
    rc::Rc,
    cell::RefCell,
    mem::MaybeUninit,
    ptr::addr_of,
    os::raw::{c_void, c_int}
};

#[derive(Debug)]
pub struct ConvolutionOp {
    input_tensor: InputTensor,
    output_tensor: OutputTensor,
    context: Rc<cudnnHandle_t>,

    filter_desc: cudnnFilterDescriptor_t,
    filter_data: DevicePtr,

    conv_desc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    
    workspace: DevicePtr,
    
    scales: Scale

}

// TODO: Move bindings to place of use
impl ConvolutionOp {
    pub fn new(context: Rc<cudnnHandle_t>,
        input_tensor: InputTensor,
        output_tensor: OutputTensor,
        data_type: &cudnnDataType,
        filters: usize,
        input_channels: usize,
        size_y: usize,
        size_x: usize,
        pad_y: usize,
        pad_x: usize,
        stride_y: usize,
        stride_x: usize,
        weights: Option<&Vec<f32>>
    ) -> Result<ConvolutionOp, RuntimeError> {
        // # TODO: Fix descriptor leaks
        let filter_desc = cudnnCreateFilterDescriptor().map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;
        cudnnSetFilter4dDescriptor(filter_desc, data_type.clone(), filters, input_channels, size_y, size_x).map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;

        let conv_desc = cudnnCreateConvolutionDescriptor().map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;
        cudnnSetConvolution2dDescriptor(conv_desc, pad_y, pad_x, stride_y, stride_x, 1, 1, data_type.clone()).map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;

        unsafe {
            let res = pnn_sys::cudnnSetConvolutionMathType(conv_desc, pnn_sys::cudnnMathType_t_CUDNN_TENSOR_OP_MATH);
            if res != 0 {
                return Err(RuntimeError::Cudnn(cudnnError::from(res)));
            }
        }

        unsafe {
            let n: c_int = 0;
            let c: c_int = 0;
            let h: c_int = 0;
            let w: c_int = 0;
            let ret = pnn_sys::cudnnGetConvolution2dForwardOutputDim(
                conv_desc,
                input_tensor.borrow().desc(),
                filter_desc,
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

        use pnn_sys::{
            cudnnConvolutionFwdAlgoPerf_t,
            cudnnFindConvolutionForwardAlgorithm,
            cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            cudnnGetConvolutionForwardWorkspaceSize
        };

        let mut algo: cudnnConvolutionFwdAlgo_t = cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        unsafe {
            let perf_result: cudnnConvolutionFwdAlgoPerf_t = MaybeUninit::zeroed().assume_init();
            let n_results: c_int = 0;
            let ret = cudnnFindConvolutionForwardAlgorithm(
                *context.as_ref(),
                input_tensor.borrow().desc(),
                filter_desc,
                conv_desc,
                output_tensor.borrow().desc(),
                1, // Query only fastest
                addr_of!(n_results) as *mut c_int,
                addr_of!(perf_result) as *mut cudnnConvolutionFwdAlgoPerf_t
            );
            if ret != 0 {
                return Err(RuntimeError::Cudnn(cudnnError::from(ret)));
            }
            if n_results != 1 {
                return Err(RuntimeError::Other(String::from("Cann't find faster CNN algorithm")))
            }
            algo = perf_result.algo;
        }

        let workspace;
        unsafe {
            let workspace_size: usize = 0;
            let ret = cudnnGetConvolutionForwardWorkspaceSize(
                *context.as_ref(),
                input_tensor.borrow().desc(),
                filter_desc,
                conv_desc,
                output_tensor.borrow().desc(),
                algo,
                addr_of!(workspace_size) as *mut usize
            );
            if ret != 0 {
                return Err(RuntimeError::Cudnn(cudnnError::from(ret)));
            }
            workspace = DevicePtr::new(cudnnDataType::HALF, workspace_size / 2)?;
        }
        let scales = Scale::new(&data_type, 1., 0.);
        let mut filter_data = DevicePtr::new(data_type.clone(), input_channels * filters * size_x * size_y)?;
        if let Some(w) = weights {
            filter_data.load_with_conversion(&w)?;
        }

        Ok(ConvolutionOp{input_tensor, output_tensor, context, filter_desc, filter_data, conv_desc, algo, workspace, scales})
    }
}

impl LayerOp for ConvolutionOp {
    
    fn forward(&mut self) -> Result<(), RuntimeError> {
        use pnn_sys::{
            cudnnConvolutionForward
        };
        unsafe {
            let x_desc;
            let x_ptr;
            {   // Allow inplace operations for layer
                x_desc = self.input_tensor.borrow().desc();
                x_ptr = self.input_tensor.borrow_mut().ptr().borrow().ptr();
            }
            let mut y = self.output_tensor.borrow_mut();
            let ret = cudnnConvolutionForward(
                *self.context.as_ref(),
                self.scales.alpha_ptr(),
                x_desc,
                x_ptr,
                self.filter_desc,
                self.filter_data.ptr(),
                self.conv_desc,
                self.algo,
                self.workspace.ptr() as *mut c_void,
                self.workspace.size(),
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

impl Drop for ConvolutionOp {
    fn drop(&mut self) {
        cudnnDestroyFilterDescriptor(self.filter_desc).expect("Could free filter descriptor");
        cudnnDestroyConvolutionDescriptor(self.conv_desc).expect("Could free conv descriptor");
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
        let x_data = Rc::new(RefCell::new(DevicePtr::new(dtype.clone(), 4 * 3 * 416 * 416).unwrap()));
        let inp = Rc::new(RefCell::new(Tensor::new(Box::new(LayerShape::from_nchw(4, 3, 416, 416)), x_data.clone()).unwrap()));
    
        let y_data = Rc::new(RefCell::new(DevicePtr::new(dtype.clone(), 4 * 32 * 416 * 416).unwrap()));
        let outp = Rc::new(RefCell::new(Tensor::new(Box::new(LayerShape::from_nchw(4, 32, 416, 416)), y_data.clone()).unwrap()));
    
        let handle = Rc::new(cudnnCreate().unwrap());
    
        let mut conv = ConvolutionOp::new(
            handle,
            inp,
            outp,
            &dtype,
            32, 3,
            3, 3,
            1, 1,
            1, 1,
            None
        ).unwrap();
    
        conv.forward().unwrap();
    }
}