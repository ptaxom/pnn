use std::{fmt,
    error::Error,
    self};
use crate::nn::BuildError;

#[derive(Debug)]
pub struct DeserializationError(pub String);

impl fmt::Display for DeserializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for DeserializationError {}

#[derive(Debug)]
pub struct ParseError(pub String);

impl ParseError {
    pub fn generate<T>(err: &str) -> Result<T, BuildError> {
        Err(BuildError::Parse(ParseError(String::from("Couldnt parse line ") + err)))
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ParseError {}