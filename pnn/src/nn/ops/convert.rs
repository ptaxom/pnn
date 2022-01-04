use crate::cudnn::{
    cudnnHandle_t,
    cudnnDataType,
    cudaStream_t,
    cudnnGetStream,
    cvt_data
};
use crate::nn::{LayerOp, RuntimeError, InputTensor, OutputTensor};


use std::{
    rc::Rc,
    os::raw::{c_void}
};



#[derive(Debug)]
pub struct ConvertOp {
    input_tensor: InputTensor,
    output_tensor: OutputTensor,
    stream: cudaStream_t,
}

const SUPPORTED_DTYPES: [cudnnDataType; 3] = [
    cudnnDataType::HALF,
    cudnnDataType::FLOAT,
    cudnnDataType::DOUBLE
];

impl ConvertOp {
    pub fn new(context: Rc<cudnnHandle_t>,
        input_tensor: InputTensor,
        output_tensor: OutputTensor
    ) -> Result<ConvertOp, RuntimeError> {
        if !SUPPORTED_DTYPES.contains(&output_tensor.borrow().data_type()) {
            return Err(RuntimeError::Other(String::from("Not supported type for convert")))
        }
        if input_tensor.borrow().shape().size() !=
            output_tensor.borrow().shape().size()
             {
                return Err(RuntimeError::Other(String::from("Mismatched tensor shape")))
            }
        let stream = cudnnGetStream(*context.as_ref()).map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;
        Ok(ConvertOp{input_tensor, output_tensor, stream})
    }
}

impl LayerOp for ConvertOp {
    
    fn forward(&mut self) -> Result<(), RuntimeError> {
        let capacity;
        let iptr;
        let idtype;
        let optr;
        let odtype;
        {
            let mut inp_tensor = self.input_tensor.borrow_mut();
            let ptr = inp_tensor.ptr();
            capacity = ptr.borrow().capacity();
            iptr = ptr.borrow_mut().ptr() as *mut c_void;
            idtype = inp_tensor.data_type();
        }
        {
            let mut out_tensor = self.output_tensor.borrow_mut();
            optr = out_tensor.ptr().borrow_mut().ptr() as *mut c_void;
            odtype = out_tensor.data_type();
        }
        cvt_data(
            optr,
            iptr,
            capacity,
            odtype,
            idtype,
            self.stream
        ).map_err(|e| {
            RuntimeError::Cuda(e)
        })?;
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
        use std::cell::RefCell;

        let xdtype = cudnnDataType::FLOAT;
        let ydtype = cudnnDataType::HALF;
        let x_data = Rc::new(RefCell::new(DevicePtr::new(xdtype.clone(), 4 * 512 * 16 * 16).unwrap()));
        let inp = Rc::new(RefCell::new(Tensor::new(Box::new(LayerShape::from_nchw(4, 512, 16, 16)), x_data.clone()).unwrap()));

        let y_data = Rc::new(RefCell::new(DevicePtr::new(ydtype.clone(), 4 * 512 * 16 * 16).unwrap()));
        let outp = Rc::new(RefCell::new(Tensor::new(Box::new(LayerShape::from_nchw(4, 512, 16, 16)), y_data.clone()).unwrap()));
    
        let handle = Rc::new(cudnnCreate().unwrap());
        let mut conv = ConvertOp::new(
            handle.clone(),
            inp,
            outp
        ).unwrap();
        conv.forward().unwrap();
    }
}