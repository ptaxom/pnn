#![allow(non_snake_case)]

use std::{
    fmt,
    self,
    any::Any
};
use core::fmt::Debug;
use crate::nn::errors::*;

pub trait Shape: fmt::Display {
    fn N(&self) -> usize;

    fn C(&self) -> usize;

    fn H(&self) -> Option<usize>;

    fn W(&self) -> Option<usize>;

    fn concat(&self, other: &dyn Shape, axis: usize) -> Result<Box<dyn Shape> , ShapeError>;

    fn dims(&self) -> Vec<usize>;

    fn as_any(&self) -> &dyn Any;

    fn size(&self) -> usize;
}

impl Debug for dyn Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let dims_str: Vec<String> = self.dims().iter().map(|d| {d.to_string()}).collect();
        write!(f, "{}", dims_str.join("x"))
    }
}

// Representing shape of layer
#[derive(Debug, Clone)]
pub struct LayerShape {
    // Storing shape of layer in format NCHW, where
    // N - batch size,
    // C - channels or filters(features) size
    // H - height
    // W - width

    // packed into vec
    dims: Vec<usize>,
    _size: usize
}

impl LayerShape {
    pub fn new(dims: Vec<usize>) -> Self{
        assert!(dims.len() > 1 && dims.len() < 5);
        let _size: usize = dims.iter().product::<usize>();
        Self{dims, _size} // TODO: Fix it
    }

    pub fn from_nc(n: usize, c: usize) -> Self {
        Self{dims: vec![n, c], _size: n * c}
    }

    pub fn from_nch(n: usize, c: usize, h: usize) -> Self {
        Self{dims: vec![n, c, h], _size: n * c * h}
    }

    pub fn from_nchw(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self{dims: vec![n, c, h, w], _size: n * c * h * w}
    }
}

impl Shape for LayerShape {

    fn N(&self) -> usize {
        *(self.dims.get(0).unwrap()) // ?????????
    }

    fn C(&self) -> usize {
        *(self.dims.get(1).unwrap()) // ?????????
    }

    fn H(&self) -> Option<usize> {
        match self.dims.get(2) {
            Some(d) => Some(*d),
            None => None
        }
    }

    fn W(&self) -> Option<usize> {
        match self.dims.get(3) {
            Some(d) => Some(*d),
            None => None
        }
    }

    fn dims(&self) -> Vec<usize> {
        self.dims.clone()
    }

    fn concat(&self, other: &dyn Shape, axis: usize) -> Result<Box<dyn Shape> , ShapeError> {
        if axis == 0 {
            return Err(ShapeError(String::from("Couldnt concat shapes by batch size")))
        }
        if self.dims.len() != other.dims().len(){
            return Err(ShapeError(String::from("Couldnt concat shapes with different size")))
        }
        let mut new_dims = vec![self.dims[0]];
        for i in 1..self.dims.len() {
            let dim;
            if i == axis {
                dim = self.dims[i] + other.dims()[i]
            } else if self.dims[i] == other.dims()[i] {
                dim = self.dims[i]
            } else {
                return  Err(ShapeError(format!("Couldnt concat across axis {}", i)))
            }
            new_dims.push(dim);
        }
        Ok(Box::new(LayerShape::new(new_dims)))
    }

    fn as_any(&self) -> &dyn Any{
        self
    }

    fn size(&self) -> usize {
        self._size
    }
}

impl PartialEq for LayerShape {
    fn eq(&self, other: &Self) -> bool {
        self.dims() == other.dims()
    }
}

impl fmt::Display for LayerShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims_as_str: Vec<String> = self.dims.iter().map(|x| {x.to_string()}).collect();
        write!(f, "LayerShape[{}]", dims_as_str.join("x"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_created() {
        let shape = LayerShape::new(vec![1, 2, 3]);
        assert_eq!(shape.dims, [1, 2, 3]);
    }

    #[test]
    fn shape_created2() {
        let shape = LayerShape::new(vec![1, 2, 3]);
        assert_ne!(shape.dims, [1, 2, 4]);
    }

    #[test]
    #[should_panic]
    fn shape_ncreated() {
        let _shape = LayerShape::new(vec![1]);
    }

    #[test]
    #[should_panic]
    fn shape_ncreated2() {
        let _shape = LayerShape::new(vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn shapes_eq() {
        let shape1 = LayerShape::from_nch(5, 6, 7);
        let shape2 = LayerShape::new(vec![5, 6, 7]);
        assert_eq!(shape1, shape2);
    }

    #[test]
    fn shapes_eq2() {
        let shape1 = LayerShape::from_nc(5, 11);
        let shape2 = LayerShape::new(vec![5, 11]);
        assert_eq!(shape1, shape2);
    }

    #[test]
    fn shapes_neq() {
        let shape1 = LayerShape::from_nch(5, 8, 7);
        let shape2 = LayerShape::new(vec![5, 6, 7]);
        assert_ne!(shape1, shape2);
    }

    #[test]
    fn shapes_neq2() {
        let shape1 = LayerShape::from_nch(5, 8, 7);
        let shape2 = LayerShape::new(vec![5, 6, 7, 8]);
        assert_ne!(shape1, shape2);
    }

    #[test]
    fn get_n() {
        let shape = LayerShape::from_nch(5, 8, 7);
        assert_eq!(shape.N(), 5)
    }

    #[test]
    fn get_n_fail() {
        let shape = LayerShape::from_nc(6, 8);
        assert_ne!(shape.N(), 5)
    }

    #[test]
    fn get_c() {
        let shape = LayerShape::from_nch(5, 8, 7);
        assert_eq!(shape.C(), 8)
    }

    #[test]
    fn get_c_fail() {
        let shape = LayerShape::from_nc(6, 8);
        assert_ne!(shape.C(), 7)
    }

    #[test]
    fn get_h() {
        let shape = LayerShape::from_nch(5, 8, 7);
        assert_eq!(shape.H(), Some(7));
    }

    #[test]
    fn get_h_fail() {
        let shape = LayerShape::from_nc(6, 8);
        assert_eq!(shape.H(), None)
    }

    #[test]
    fn get_w() {
        let shape = LayerShape::from_nchw(5, 8, 7, 9);
        assert_eq!(shape.W(), Some(9))
    }

    #[test]
    fn get_w_fail() {
        let shape = LayerShape::from_nch(6, 8, 2);
        assert_eq!(shape.W(), None)
    }

    #[test]
    #[should_panic(expected = "Couldnt concat shapes with different size")]
    fn concat_mismatch() {
        let shape1 = LayerShape::from_nch(5, 8, 7);
        let shape2 = LayerShape::new(vec![5, 8, 7, 8]);
        shape1.concat(&shape2, 2).unwrap();
    }

    #[test]
    #[should_panic(expected = "Couldnt concat shapes by batch size")]
    fn concat_batchsize() {
        let shape1 = LayerShape::from_nch(5, 8, 7);
        let shape2 = LayerShape::new(vec![5, 6, 7, 8]);
        shape1.concat(&shape2, 0).unwrap();
    }

    #[test]
    fn concat_correct() {
        let shape1 = LayerShape::from_nchw(32, 128, 64, 64);
        let shape2 = LayerShape::new(vec![32, 32, 64, 64]);
        let target = LayerShape::new(vec![32, 160, 64, 64]);
        let result = shape1.concat(&shape2, 1).unwrap();
        let res2 = result.as_any().downcast_ref::<LayerShape>().unwrap();
        assert_eq!(*res2, target);
    }

    #[test]
    #[should_panic(expected = "Couldnt concat across axis 1")]
    fn concat_mismatch_axis1() {
        let shape1 = LayerShape::from_nchw(32, 128, 64, 64);
        let shape2 = LayerShape::new(vec![32, 32, 64, 64]);
        shape1.concat(&shape2, 2).unwrap();
    }

    #[test]
    #[should_panic(expected = "Couldnt concat across axis 2")]
    fn concat_mismatch_axis2() {
        let shape1 = LayerShape::from_nchw(32, 32, 64, 64);
        let shape2 = LayerShape::new(vec![32, 33, 65, 64]);
        shape1.concat(&shape2, 1).unwrap();
    }

    #[test]
    #[should_panic(expected = "Couldnt concat across axis 3")]
    fn concat_mismatch_axis3() {
        let shape1 = LayerShape::from_nchw(32, 32, 64, 65);
        let shape2 = LayerShape::new(vec![32, 33, 64, 64]);
        shape1.concat(&shape2, 1).unwrap();
    }

    #[test]
    fn test_display() {
        let shape = LayerShape::from_nchw(32, 256, 7, 7);
        assert_eq!(shape.to_string(), "LayerShape[32x256x7x7]");
    }
}