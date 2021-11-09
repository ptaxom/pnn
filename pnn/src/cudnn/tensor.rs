use crate::nn::{Shape, ShapeError};
use std::{os::raw::c_void, fmt, error::Error};
use pnn_sys::{cudnnTensorDescriptor_t};
use crate::cudnn::*;

pub type HostDataType = Option<Vec<f32>>;


//Container to store device ptr and tensor information in conjunction 
pub struct Tensor {
    // Shape of tensor in pnn type
    pub shape: Box<dyn Shape>,
    // Pointer to GPU memory
    device_data_ptr: *mut c_void,
    // Tensor descriptor
    tensor_desc: cudnnTensorDescriptor_t,
    // Optional reflection of GPU memory to host. Evaluating in lazy format
    host_data: HostDataType
}

impl Tensor {
    pub fn new(shape: Box<dyn Shape>) -> Result<Self, Box<dyn std::error::Error>> {
        // allocate CUDA memory at GPU
        // Currently support only for f32
        let device_data_ptr = cudaMalloc(std::mem::size_of::<f32>() * shape.size())?;
        let tensor_desc = cudnnCreateTensorDescriptor()?;
        cudnnSetTensor4dDescriptor(
            tensor_desc,
            cudnnDataType::FLOAT,
            shape.N(),
            shape.C(),
            shape.H().unwrap_or(1),
            shape.W().unwrap_or(1)
        )?;
        let host_data: HostDataType  = None;
        Ok(Self{shape, device_data_ptr, tensor_desc, host_data})
    }

    fn init_host_data(&mut self) {
        if self.host_data == None {
            self.host_data = Some(Vec::with_capacity(self.shape.size()));
        }
        match &mut self.host_data {
            Some(data) => unsafe { data.set_len(self.shape.size())},
            None => ()
        };
    }

    fn sync_with_gpu(&mut self) -> Result<(), cudaError> {
        self.init_host_data();
        match &mut self.host_data {
            Some(data) => cudaMemcpy(data.as_mut_ptr() as *mut c_void, self.device_data_ptr, self.shape.size() * std::mem::size_of::<f32>(), cudaMemcpyKind::DeviceToHost)?,
            None => ()
        };
        Ok(())
    }

    fn sync_with_host(&mut self) -> Result<(), cudaError> {
        self.init_host_data();
        match &mut self.host_data {
            Some(data) => cudaMemcpy(self.device_data_ptr, data.as_mut_ptr() as *mut c_void, self.shape.size() * std::mem::size_of::<f32>(), cudaMemcpyKind::HostToDevice)?,
            None => ()
        };
        Ok(())
    }

    pub fn load(&mut self, other: &Vec<f32>) -> Result<(), Box<dyn Error>> {
        // #TODO: copy host -> host, and then host -> device. rework
        if self.shape.size() != other.len() {
            return Err(Box::new(ShapeError{description: String::from("Couldnt load from array with wrong size")}))
        }
        self.host_data = Some(other.clone());
        self.sync_with_host()?;
        Ok(())
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        cudaFree(self.device_data_ptr).expect("Couldnt free allocated CUDA memory");
        cudnnDestroyTensorDescriptor(self.tensor_desc).expect("Could free tensor descriptor");
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // #TODO fix it with lazy evaltate
        let len = self.shape.size();
        let mut data: Vec<f32> = Vec::with_capacity(len);
        unsafe { data.set_len(len); }
        cudaMemcpy(data.as_mut_ptr() as *mut c_void, self.device_data_ptr, self.shape.size() * std::mem::size_of::<f32>(), cudaMemcpyKind::DeviceToHost)
            .expect("Couldnt copy data to CPU during Tensor Display");

        let content: Vec<String> = data.iter().map(|x| {(*x).to_string()}).collect();
        write!(f, "Tensor with {}. Data: [{}]", self.shape, content.join("x"))
    }
}

pub fn add(handle: cudnnHandle_t, a: &Tensor, b: &mut Tensor) -> Result<(), cudnnError> {
    cudnnAddTensor(handle, 1., a.tensor_desc, a.device_data_ptr, 1., b.tensor_desc, b.device_data_ptr)?;
    Ok(())
}