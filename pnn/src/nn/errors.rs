use std::{
    fmt,
    error::Error
};
use crate::cudnn::{cudaError, cudnnError};

#[derive(Debug)]
pub enum BuildError {
    DimInferError(ShapeError),
    Rebuild,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::DimInferError(e) => write!(f, "{}", e),
            BuildError::Rebuild => write!(f, "Network already builded"),
            _ => write!(f, "Unknown BuildError"),
        }
    }
}

impl Error for BuildError {}


#[derive(Debug)]
pub struct ShapeError(pub String);

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ShapeError {}

#[derive(Debug)]
pub enum RuntimeError {
    Cuda(cudaError),
    Cudnn(cudnnError),
    Other(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::Cuda(e) => write!(f, "{}", e),
            RuntimeError::Cudnn(e) => write!(f, "{}", e),
            RuntimeError::Other(e) => write!(f, "{}", e),
            _ => write!(f, "Other runtime error"),
        }
    }
}

impl Error for RuntimeError {}