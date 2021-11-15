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

        use pnn_sys::{
            cudnnConvolutionFwdAlgoPerf_t,
            cudnnFindConvolutionForwardAlgorithm,
            cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            cudnnGetConvolutionForwardWorkspaceSize
        };

        let mut algo: cudnnConvolutionFwdAlgo_t = cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        unsafe {
            let mut perf_result: cudnnConvolutionFwdAlgoPerf_t = MaybeUninit::zeroed().assume_init();
            let mut n_results: c_int = 0;
            let ret = cudnnFindConvolutionForwardAlgorithm(
                *context.as_ref(),
                input_tensor.as_ref().borrow_mut().desc(),
                filter_desc,
                conv_desc,
                output_tensor.as_ref().borrow_mut().desc(),
                1, // Query only fastest
                addr_of!(n_results) as (*mut c_int),
                addr_of!(perf_result) as (*mut cudnnConvolutionFwdAlgoPerf_t)
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
            let mut workspace_size: usize = 0;
            let ret = cudnnGetConvolutionForwardWorkspaceSize(
                *context.as_ref(),
                input_tensor.as_ref().borrow_mut().desc(),
                filter_desc,
                conv_desc,
                output_tensor.as_ref().borrow_mut().desc(),
                algo,
                addr_of!(workspace_size) as (*mut usize)
            );
            if ret != 0 {
                return Err(RuntimeError::Cudnn(cudnnError::from(ret)));
            }
            workspace = DevicePtr::new(cudnnDataType::HALF, workspace_size / 2)?;
        }
        let scales = Scale::new(&data_type, 1., 0.);


        Ok(ConvolutionOp{input_tensor, output_tensor, context, filter_desc, filter_data, conv_desc, algo, workspace, scales})
    }
}

impl LayerOp for ConvolutionOp {
    
    fn forward(&mut self) -> Result<(), RuntimeError> {
        use pnn_sys::{
            cudnnConvolutionForward
        };
        unsafe {
            let mut x = self.input_tensor.borrow_mut();
            let mut y = self.output_tensor.borrow_mut();
            let ret = cudnnConvolutionForward(
                *self.context.as_ref(),
                self.scales.alpha_ptr(),
                x.desc(),
                x.ptr().borrow().ptr(),
                self.filter_desc,
                self.filter_data.ptr(),
                self.conv_desc,
                self.algo,
                self.workspace.ptr() as *mut c_void,
                self.workspace.size(),
                self.scales.beta_ptr(),
                y.desc(),
                y.ptr().borrow_mut().ptr() as *mut c_void
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