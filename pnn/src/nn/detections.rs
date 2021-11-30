use std::{
    fmt
};

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct BoundingBox{
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    class_id: usize,
    probability: f32,
    objectness: f32
}


fn max(a: f32, b: f32) -> f32 {
    if a > b {return a} else {return b}
}

fn min(a: f32, b: f32) -> f32 {
    if a < b {return a} else {return b}
}


impl fmt::Display for BoundingBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // write!(f, "Bbox [({:.2},{:.2})->({:.2},{:.2})], Class={}", self.x0, self.y0, self.x1, self.y1, self.class_id)
        write!(f, "[{:.2},{:.2}, {:.2},{:.2}],", self.x0, self.y0, self.x1, self.y1)
    }
}

impl BoundingBox {
    pub fn new(x0: f32,
            y0: f32,
            x1: f32,
            y1: f32,
            class_id: usize,
            probability: f32,
            objectness: f32) -> BoundingBox {
        BoundingBox{x0, y0, x1, y1, class_id, probability, objectness}
    }
    pub fn area(&self) -> f32 {
        (self.x1 - self.x0) * (self.y1 - self.y0)
    }

    pub fn iou(&self, other: &BoundingBox) -> f32 {
        if self.class_id != other.class_id {
            return 0.
        }

        let x0 = max(self.x0, other.x0);
        let y0 = max(self.y0, other.y0);
        let x1 = min(self.x1, other.x1);
        let y1 = min(self.y1, other.y1);
        
        let w = max(x1 - x0, 0.);
        let h = max(y1 - y0, 0.);
        let union_area = w * h;
        return union_area / (self.area() + other.area() - union_area + 0.000001);
    }

    pub fn nms(bboxes: &Vec<BoundingBox>, iou_tresh: f32) -> Vec<BoundingBox> {
        let mut id = 0;
        let mut boxes = bboxes.clone();
        boxes.sort_by(|a, b| {
            b.objectness.partial_cmp(&a.objectness).unwrap()
        });

        while id < boxes.len() {
            boxes = boxes.iter().enumerate().filter_map(|(i, b)| {
                if i <= id || boxes[id].iou(b) < iou_tresh {
                    return Some((i, b))
                }
                None
            }).map(|s| {*s.1}).collect();
            id += 1;
        }
        boxes
    }

}