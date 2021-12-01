use crate::nn::{DetectionsParser, BoundingBox, RuntimeError};
use crate::cudnn::{DevicePtr};

use std::{
    rc::Rc,
    cell::RefCell
};

pub struct YoloHeadParser {
    height: usize,
    width: usize,
    channels: usize,
    batchsize: usize,
    n_classes: usize,
    scale: f32,
    anchors: Vec<(f32, f32)>,

    binding: Rc<RefCell<DevicePtr>>

}

impl YoloHeadParser {
    pub fn new(height: usize,
        width: usize,
        channels: usize,
        batchsize: usize,
        n_classes: usize,
        scale: f32,
        anchors: Vec<(f32, f32)>,
        binding: Rc<RefCell<DevicePtr>>) -> YoloHeadParser {
            YoloHeadParser{width, height, batchsize, n_classes, scale, anchors, binding, channels}
        }
}

fn max(a: f32, b: f32) -> f32 {
    if a > b {return a} else {return b}
}

fn min(a: f32, b: f32) -> f32 {
    if a < b {return a} else {return b}
}

impl DetectionsParser for YoloHeadParser {
    fn get_bboxes(&self, threshold: f32) -> Result<Vec<Vec<BoundingBox>>, RuntimeError>{
        let data = self.binding.borrow().download::<f32>()?;
        let mut batch_predictions = Vec::new();

       
        let delta = -0.5 * (self.scale - 1.);
        let stride = self.width * self.height;

        for batch_id in 0..self.batchsize {
            let mut sample_bboxes: Vec<BoundingBox> = Vec::new();

            for head_id in 0..self.anchors.len(){
                for i in 0..self.height {
                    for j in 0..self.width {
                        let index: usize = batch_id * self.channels * stride +
                                        head_id * (self.n_classes + 4 + 1) * stride +
                                        i * self.width +
                                        j;

                        let objectness = data[index + 4 * stride];
                        if objectness > threshold {
                            let x_c = (j as f32 + data[index + 0 * stride] * self.scale + delta) / self.width  as f32;
                            let y_c = (i as f32 + data[index + 1 * stride] * self.scale + delta) / self.height as f32;
                            
                            let w = 2. * data[index + 2 * stride];
                            let w = w * w * self.anchors[head_id].0;

                            let h = 2. * data[index + 3 * stride];
                            let h = h * h * self.anchors[head_id].1;

                            let x0 = max(x_c - w / 2., 0.);
                            let y0 = max(y_c - h / 2., 0.);
                            let x1 = min(x_c + w / 2., 1.);
                            let y1 = min(y_c + h / 2., 1.);

                            let mut class_id = self.n_classes + 1;
                            let mut probability = -1.;
                            for cls_id in 0..self.n_classes {
                                let prob = data[index + (5 + cls_id) * stride] * objectness;
                                if  prob > threshold && prob > probability {
                                    probability = prob;
                                    class_id = cls_id;
                                }
                            }
                            if class_id != self.n_classes + 1 {
                                sample_bboxes.push(
                                    BoundingBox::new(x0, y0, x1, y1, class_id, objectness, probability)
                                )
                            }
                        }
                    }
                }
            }
            batch_predictions.push(sample_bboxes);
        }
        Ok(batch_predictions)
    }
}