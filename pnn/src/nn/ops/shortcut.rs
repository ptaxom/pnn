use crate::cudnn::{Tensor,
    cudnnHandle_t,
    cudnnDataType,
    cudnnError,
    Scale
};
use crate::nn::{LayerOp, RuntimeError, InputTensor, OutputTensor};


use std::{
    rc::Rc,
    cell::RefCell,
    os::raw::{c_void}
};

#[derive(Debug)]
pub struct ShortcutOp {
    input_tensors: Vec<InputTensor>,
    output_tensor: OutputTensor,
    context: Rc<cudnnHandle_t>,
    
    scale: Scale
}


impl ShortcutOp {
    pub fn new(context: Rc<cudnnHandle_t>,
        input_tensors: Vec<InputTensor>,
        output_tensor: OutputTensor
    ) -> Result<ShortcutOp, RuntimeError> {
        let scale = Scale::new(&output_tensor.borrow().data_type(), 1., 1.);
        Ok(ShortcutOp{input_tensors, output_tensor,
            context, scale})
    }
}

impl LayerOp for ShortcutOp {
    
    fn forward(&mut self) -> Result<(), RuntimeError> {
        let mut target = self.output_tensor.borrow_mut();
        for inp in &mut self.input_tensors {
            unsafe {
                let mut y = inp.borrow_mut();
                let res = pnn_sys::cudnnAddTensor(
                    *self.context.as_ref(),
                    self.scale.alpha_ptr()  as *const c_void,
                    y.desc(),
                    y.ptr().borrow().ptr(),
                    self.scale.alpha_ptr()  as *const c_void,
                    target.desc(),
                    target.ptr().borrow().ptr() as *mut c_void
                );
                if res != 0 {
                    return Err(RuntimeError::Cudnn(cudnnError::from(res)))
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

        let y_data = Rc::new(RefCell::new(DevicePtr::new(dtype.clone(), 4 * 512 * 16 * 16).unwrap()));
        let outp = Rc::new(RefCell::new(Tensor::new(Box::new(LayerShape::from_nchw(4, 512, 16, 16)), y_data.clone()).unwrap()));
    
        let handle = Rc::new(cudnnCreate().unwrap());
        let mut route = ShortcutOp::new(
           handle,
            vec![inp.clone(),inp.clone(), inp.clone(), inp.clone()],
            outp
        ).unwrap();
        route.forward().unwrap();
    }
}