use std::{
    os::raw::{c_void, c_double, c_float}
};
use crate::cudnn::cudnnDataType;

// #TODO: implement it with Box<T>
enum ScalePair {
    Single((Box<f32>, Box<f32>)),
    Double((Box<f64>, Box<f64>)),
}

pub struct Scale{
    data: ScalePair
}


impl Scale {
    pub fn new(data_type: &cudnnDataType, a: f64, b: f64) -> Scale {
        let data = match data_type {
            cudnnDataType::DOUBLE => ScalePair::Double((Box::new(a), Box::new(b))),
            _ =>  ScalePair::Single((Box::new(a as f32), Box::new(b as f32)))
        };
        Scale{data}
    }

    pub fn alpha_ptr(&self) -> *const c_void {
        match &self.data {
            ScalePair::Single(v) => (&*v.0 as *const c_float) as *const c_void,
            ScalePair::Double(v) => (&*v.0 as *const c_double) as *const c_void
        }
    }

    pub fn beta_ptr(&self) -> *const c_void {
        match &self.data {
            ScalePair::Single(v) => (&*v.1 as *const c_float) as *const c_void,
            ScalePair::Double(v) => (&*v.1 as *const c_double) as *const c_void
        }
    }
}
