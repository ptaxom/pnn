use crate::nn::shape::*;
use crate::nn::errors::*;
use crate::nn::ops::{LayerOp, OutputTensor};
use crate::cudnn::{DevicePtr, cudnnDataType};
use crate::nn::BoundingBox;

use std::{
    collections::HashMap,
    rc::Rc,
    cell::RefCell
};

pub trait DetectionsParser {
    fn get_bboxes(&self, threshold: f32) -> Result<Vec<Vec<BoundingBox>>, RuntimeError>;
}

pub trait Engine {

    fn forward(&mut self) -> Result<(), RuntimeError>;

    fn inputs(&self) -> Vec<String>;

    fn input_binding(&self, name: &String) -> Option<Rc<RefCell<DevicePtr>>>;

    fn output_binding(&self, name: &String) -> Option<Rc<RefCell<DevicePtr>>>;

    fn add_detections_parser(&mut self, binding_name: &String, parser: Box<dyn DetectionsParser>);

    fn batchsize(&self) -> usize;

    fn detection_parsers(&self) -> &HashMap<String, Box<dyn DetectionsParser>>;

    fn get_detections(&self, threshold: f32, nms_threshold: f32) -> Result<Vec<Vec<BoundingBox>>, RuntimeError> {
        let batchsize = self.batchsize();

        let mut predictions: Vec<Vec<BoundingBox>> = Vec::with_capacity(batchsize);
        predictions.resize_with(batchsize, ||{Vec::new()});

        for (_, parser) in self.detection_parsers() {
            let mut head_predictions = parser.get_bboxes(threshold)?;
            for batch_id in 0..batchsize {
                predictions[batch_id].append(&mut head_predictions[batch_id]);
            }
        }
        Ok(predictions.iter().map(|x| {BoundingBox::nms(x, nms_threshold)}).collect())
    }
}

mod yolo;
mod cudnn;
mod trt;

pub use yolo::*;
pub use cudnn::*;
pub use trt::*;