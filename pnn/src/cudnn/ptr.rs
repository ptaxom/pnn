use crate::cudnn::{cudnnDataType, cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyKind, cvt_data, cudaStream_t, cudaDeviceSynchronize};
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

    pub fn load_with_conversion<T>(&mut self, other: &Vec<T>) -> Result<(), RuntimeError> {
        if self.capacity != other.len() {
            return Err(RuntimeError::Other(String::from("Couldnt load from vector with different size")))
        }
        let vec_type = std::any::type_name::<T>();
        if  vec_type == "f64" || self.type_name == "f64" {
            return Err(RuntimeError::Other(format!("Conversions to f64 is not allowed")))
        }
        if  vec_type == self.type_name {
            return self.load(other);
        }
        let src_dtype;
        if vec_type == "f32" {
            src_dtype = cudnnDataType::FLOAT;
        } else {
            src_dtype = cudnnDataType::HALF;
        }

        let mut src_device = DevicePtr::new(src_dtype.clone(), other.len())?;
        src_device.load(other)?;
        cvt_data(
            self.ptr,
            src_device.ptr() as *mut c_void,
            self.capacity,
            self.data_type.clone(),
            src_dtype,
            0 as cudaStream_t
        ).map_err(|e| {
                RuntimeError::Cuda(e)
            })?;
        cudaDeviceSynchronize().map_err(|e| {
            RuntimeError::Cuda(e)
        })?;
        Ok(())
    }

    pub fn download_with_conversion<T>(&self) -> Result<Vec<T>, RuntimeError> {

        let vec_type = std::any::type_name::<T>();
        if  vec_type == "f64" || self.type_name == "f64" {
            return Err(RuntimeError::Other(format!("Conversions to f64 is not allowed")))
        }
        if  vec_type == self.type_name {
            return self.download::<T>();
        }
        let dst_dtype;
        if vec_type == "f32" {
            dst_dtype = cudnnDataType::FLOAT;
        } else {
            dst_dtype = cudnnDataType::HALF;
        }

        let dst_device = DevicePtr::new(dst_dtype.clone(), self.capacity)?;
        cvt_data(
            dst_device.ptr() as *mut c_void,
            self.ptr,
            self.capacity,
            dst_dtype,
            self.data_type.clone(),
            0 as cudaStream_t
        ).map_err(|e| {
                RuntimeError::Cuda(e)
            })?;
        cudaDeviceSynchronize().map_err(|e| {
            RuntimeError::Cuda(e)
        })?;
        return dst_device.download::<T>();
    }
    
    pub fn download<T>(&self) -> Result<Vec<T>, RuntimeError> {
        let vec_type = std::any::type_name::<T>();
        if  vec_type != self.type_name {
            return Err(RuntimeError::Other(format!("Rust Vec<{}> couldnt be loaded to DevicePtr<{}>", vec_type, self.type_name)))
        }
        
        let mut data: Vec<T> = Vec::with_capacity(self.capacity);
        unsafe { data.set_len(self.capacity); }

        crate::cudnn::cudaDeviceSynchronize().map_err(|e| {
            RuntimeError::Cuda(e)
        })?;
        cudaMemcpy(data.as_mut_ptr() as *mut c_void, self.ptr as *const c_void, self.size, cudaMemcpyKind::DeviceToHost).map_err(|e| {
            RuntimeError::Cuda(e)
        })?;
        Ok(data)
    }

    pub fn dump(&self, file_path: &String) -> Result<(), RuntimeError> {
        use std::io::prelude::*;

        let mut data: Vec<u8> = Vec::with_capacity(self.size);
        unsafe { data.set_len(self.size); }
        crate::cudnn::cudaDeviceSynchronize().map_err(|e| {
            RuntimeError::Cuda(e)
        })?;
        cudaMemcpy(data.as_mut_ptr() as *mut c_void, self.ptr as *const c_void, self.size, cudaMemcpyKind::DeviceToHost).map_err(|e| {
            RuntimeError::Cuda(e)
        })?;

        let mut file = std::fs::File::create(file_path).map_err(|_| {
            RuntimeError::Other(format!("Couldnt create {}", &file_path))
        })?;

        file.write_all(&data).map_err(|_| {
            RuntimeError::Other(format!("Couldnt write to file",))
        })?;
        Ok(())
    }

    pub fn load_bin(&mut self, file_path: &String) -> Result<(), RuntimeError> {
        use std::io::Read;

        let mut file = std::fs::File::open(file_path).map_err(|_| {
           RuntimeError::Other(format!("Couldnt open {}", &file_path))
        })?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(|_| {
            RuntimeError::Other(format!("Couldnt read from {}", &file_path))
        })?;
        if buffer.len() != self.size {
            return Err(RuntimeError::Other(String::from("Couldnt load from vector with different size")))
        }

        cudaMemcpy(self.ptr, buffer.as_mut_ptr() as *mut c_void, self.size, cudaMemcpyKind::HostToDevice).map_err(|e| {
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