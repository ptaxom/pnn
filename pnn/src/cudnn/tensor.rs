use crate::nn::{Shape, LayerShape};
use std::{os::raw::c_void,
    fmt,
    rc::Rc,
    cell::RefCell
};
use pnn_sys::{cudnnTensorDescriptor_t};
use crate::cudnn::*;
use crate::nn::RuntimeError;

//Container to store device ptr and tensor information in conjunction 
#[derive(Debug)]
pub struct Tensor {
    // Shape of tensor in pnn type
    shape: Box<dyn Shape>,
    // Tensor descriptor
    tensor_desc: cudnnTensorDescriptor_t,
    // On GPU Data
    ptr: Rc<RefCell<DevicePtr>>
}

impl Tensor {
    pub fn new(shape: Box<dyn Shape>, ptr: Rc<RefCell<DevicePtr>>) -> Result<Self, RuntimeError> {
        // allocate CUDA memory at GPU
        let tensor_desc = cudnnCreateTensorDescriptor().map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;
        cudnnSetTensor4dDescriptor(
            tensor_desc,
            ptr.borrow().data_type(),
            shape.N(),
            shape.C(),
            shape.H().unwrap_or(1),
            shape.W().unwrap_or(1)
        ).map_err(|e| {
            RuntimeError::Cudnn(e)
        })?;
        Ok(Self{shape, tensor_desc, ptr})
    }

    pub fn load<T>(&mut self, other: &Vec<T>) -> Result<(), RuntimeError> {
        // #TODO: copy host -> host, and then host -> device. rework
        if self.shape.size() != other.len() {
            return Err(RuntimeError::Other(String::from("Couldnt load from array with wrong size")))
        }
        self.ptr.as_ref().borrow_mut().load(other)?;
        Ok(())
    }

    pub fn desc(&self) -> cudnnTensorDescriptor_t {
        self.tensor_desc
    }

    pub fn ptr(&mut self) -> Rc<RefCell<DevicePtr>> {
        self.ptr.clone()
    }

    pub fn shape(&self) -> Box<dyn Shape> {
        Box::new(LayerShape::new(self.shape.dims().clone()))
    }

    pub fn data_type(&self) -> cudnnDataType {
        self.ptr.borrow().data_type()
    }

    pub fn download<T>(&self) -> Result<Vec<T>, RuntimeError> {
        self.ptr.borrow_mut().download::<T>()
    }

    pub fn dump(&self, file_path: &String) -> Result<(), RuntimeError> {
        self.ptr.borrow_mut().dump(file_path)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        cudnnDestroyTensorDescriptor(self.tensor_desc).expect("Could free tensor descriptor");
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // #TODO fix it with lazy evaltate
        let len = self.shape.size();
        let mut data: Vec<f32> = Vec::with_capacity(len);
        unsafe { data.set_len(len); }
        let ptr = self.ptr.as_ref().borrow();
        cudaMemcpy(data.as_mut_ptr() as *mut c_void, ptr.ptr(), ptr.size(), cudaMemcpyKind::DeviceToHost)
            .expect("Couldnt copy data to CPU during Tensor Display");

        let content: Vec<String> = data.iter().map(|x| {(*x).to_string()}).collect();
        write!(f, "Tensor with {}. Data: [{}]", self.shape, content.join("x"))
    }
}
