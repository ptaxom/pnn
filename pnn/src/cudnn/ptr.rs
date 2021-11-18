use crate::cudnn::{cudnnDataType, cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyKind};
use crate::nn::RuntimeError;
use std::{os::raw::c_void};

#[derive(Debug)]
pub struct DevicePtr {
    // Pointer to device memory
    ptr: *mut c_void,
    // Capacity of ptr in elements
    capacity: usize,
    // Data type
    data_type: cudnnDataType,
    // Size of area
    size: usize,
    // name of corresponding rust type
    type_name: &'static str
}

impl DevicePtr {
    pub fn new(data_type: cudnnDataType, capacity: usize) -> Result<DevicePtr, RuntimeError> {
        let (el_size, type_name): (usize, &str) = match data_type {
            cudnnDataType::HALF => (2, "f16"),
            cudnnDataType::FLOAT => (4, "f32"),
            cudnnDataType::DOUBLE => (8, "f64"),
            _ => return Err(RuntimeError::Other(String::from("Unsupported cudnnDataType")))
        };
        let size = el_size * capacity;
        // #TODO: Figure out bug with cudaMallocHost
        let ptr = cudaMalloc(size).map_err(|e| {
            RuntimeError::Cuda(e)
        })?;
        Ok(DevicePtr{ptr, capacity, data_type, size, type_name})
    }

    pub fn ptr(&self) -> *const c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn data_type(&self) -> cudnnDataType {
        self.data_type.clone()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn load<T>(&mut self, other: &Vec<T>) -> Result<(), RuntimeError> {
        if self.capacity != other.len() {
            return Err(RuntimeError::Other(String::from("Couldnt load from vector with different size")))
        }
        let vec_type = std::any::type_name::<T>();
        if  vec_type != self.type_name {
            return Err(RuntimeError::Other(format!("Rust Vec<{}> couldnt be loaded to DevicePtr<{}>", vec_type, self.type_name)))
        }
        
        cudaMemcpy(self.ptr,
            other.as_ptr() as *const c_void,
            self.size,
            cudaMemcpyKind::HostToDevice
        ).map_err(|e| {
                RuntimeError::Cuda(e)
            })?;
        Ok(())
    }

}

impl Drop for DevicePtr {
    fn drop(&mut self) {
        cudaFree(self.ptr).expect("Couldnt free allocated CUDA memory");
    }
}