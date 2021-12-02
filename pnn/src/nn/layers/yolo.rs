use std::{
    collections::HashMap,
    self,
    any::Any,
    sync::atomic::{Ordering},
    rc::Rc,
    cell::RefCell,
    sync::mpsc::{
        channel,
        Sender,
        Receiver
    },
    fmt

};

use crate::nn::shape::*;
use crate::nn::{Layer, LayerType, errors::*, BuildInformation, DetectionsParser, YoloHeadParser};
use crate::parsers::{DeserializationError, parse_numerical_field, ensure_positive, parse_list_field};
use crate::cudnn::{cudnnHandle_t, cudnnDataType, Tensor, DevicePtr};
use crate::nn::ops::{LayerOp, OutputTensor, ConvertOp};
use crate::nn::{CUDNNEngine, TRTBuilder, Engine};

//Yolo head layer
#[derive(Debug)]
pub struct YoloLayer {
    // Layer name
    name: String,
    // Layer shape
    shape: Option<Rc<dyn Shape>>,
    // Window stride
    classes: usize,
    // Scale for x,y coords
    scale: f32,
    // Anchors
    anchors: Vec<(usize, usize)>

}

const SUPPORTED_FIELDS: [&str; 2] = [
    "classes",
    "num"
];

impl Layer for YoloLayer {
    fn name(&self) -> String {
        self.name.to_string()
    }

    fn shape(&self) -> Option<Rc<dyn Shape>> {
        self.shape.clone()
    }

    fn propose_name() -> String {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!("Yolo_{}", COUNTER.fetch_add(1, Ordering::Relaxed))
    }
 
    fn infer_shape(&mut self, input_shapes: Vec<Rc<dyn Shape>>) -> Result<(), ShapeError> {
        if input_shapes.len() != 1 {
            return Err(ShapeError(String::from("YoloLayer must have exact one input layer")))
        }
        let input_shape = &input_shapes[0];
        if input_shape.dims().len() != 4 {
            return Err(ShapeError(String::from("YoloLayer can be connected only with layer, which produce 4D Tensor with format NCHW")))
        }

        self.shape = Some(Rc::new(LayerShape::from_nchw(input_shape.N(),
            self.anchors.len() * (5 + self.classes),
            input_shape.H().unwrap(), 
            input_shape.W().unwrap()
        )));
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn from_config(config: HashMap<String, String>) -> Result<Box<dyn Layer>, DeserializationError> {
        let proposed_name = YoloLayer::propose_name();
        let name: String = config.get("name").unwrap_or(&proposed_name).to_string();
        let shape = None;

        let classes = parse_numerical_field::<usize>(&config, "classes", true, None)?.unwrap();
        ensure_positive(classes, "classes", "YoloLayer")?;

        let scale = parse_numerical_field::<f32>(&config, "scale_x_y", true, None)?.unwrap();

        let anchor_sizes = parse_list_field::<usize>(&config, "anchors", "YoloLayer")?;
        let all_anchors: Vec<(usize, usize)> = anchor_sizes.chunks(2).map(|x| {
            (x[0], x[1])
        }).collect();

        let masks =  parse_list_field::<usize>(&config, "mask", "YoloLayer")?;
        let mut anchors = Vec::new();
        for (index, anchor) in all_anchors.iter().enumerate() {
            if masks.contains(&index) {
                anchors.push(anchor.clone());
            }
        }

        let _ = config.keys().filter(|k| {
            !SUPPORTED_FIELDS.contains(&&k[..])
        }).map(|k| {
            log::warn!("Not supported darknet field during deserialization of '{}'. Field '{}' not recognized", name, k)
        });

        Ok(Box::new(YoloLayer{name, shape,
            classes, anchors, scale
        }))
    }

    fn ltype(&self) -> LayerType {
        LayerType::Yolo
    }

    fn build_cudnn(&mut self, 
        engine: Rc<RefCell<CUDNNEngine>>,
        indeces: Vec<usize>,
        has_depend_layers: bool
    ) -> Result<(), BuildError> {
        let reusable = !has_depend_layers;
        let info: Vec<BuildInformation> = indeces.iter().map(|x| {
            engine.borrow().get_info(*x)
        }).collect();
        let mut tensor = info[0].tensor.clone();
        let data_type = engine.borrow().dtype();
        let mut operations: Vec<Box<dyn LayerOp>> = Vec::new();

        if data_type != cudnnDataType::FLOAT {
            let shape = self.shape().unwrap();
            let ptr = Rc::new(RefCell::new(
                DevicePtr::new(cudnnDataType::FLOAT, shape.size()).map_err(|e| {
                    BuildError::Runtime(e)
                })?
            ));

            let tensor_shape: Box<dyn Shape> = Box::new(LayerShape::new(shape.dims()));
            tensor = Rc::new(RefCell::new(
                Tensor::new(tensor_shape, ptr).map_err(|e| {
                    BuildError::Runtime(e)
                })?
            ));

            operations.push(
                Box::new(ConvertOp::new(
                    engine.borrow().context(),
                    info[0].tensor.clone(),
                    tensor.clone(),
                ).map_err(|e| {
                    BuildError::Runtime(e)
                })?)
            );
        }

        let ptr = tensor.clone().borrow_mut().ptr();
        engine.borrow_mut().add_layer(operations, BuildInformation{tensor, reusable});
        let name = self.name();
        engine.borrow_mut().add_output(&name, ptr.clone());
        let inp_size = engine.borrow().input_size();
        engine.borrow_mut().add_detections_parser(&name, self.get_parser(inp_size, ptr.clone()));

        Ok(())
    }

    fn build_trt(&mut self, 
        engine: Rc<RefCell<TRTBuilder>>,
        indeces: Vec<usize>
    ) -> Result<(), BuildError> {
        let mut engine = engine.borrow_mut();
        let id: usize = engine.last_op_id(indeces[0]);
        engine.add_yolo(id, &self.name())?;
        // TODO: check it
        engine.finilize_layer(usize::MAX);
        Ok(())
    }

}

impl YoloLayer {
    pub fn get_parser(&self, input_size: (usize, usize), ptr: Rc<RefCell<DevicePtr>>) -> Box<dyn DetectionsParser> {
        let shape = self.shape().unwrap();
        Box::new(
            YoloHeadParser::new(
                shape.H().unwrap(),
                shape.W().unwrap(),
                shape.C(),
                shape.N(),
                self.classes,
                self.scale,
                self.anchors.iter().map(|x| {
                    let (a_w, a_h) = *x;
                    let (i_h, i_w) = input_size;
                    (a_w as f32 / i_w as f32, a_h as f32 / i_h as f32)
                }).collect(),
                ptr
            )
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_config() -> HashMap<String, String> {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("classes"), String::from("80"));
        config.insert(String::from("scale_x_y"), String::from("2."));
        config.insert(String::from("anchors"), String::from("12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401"));
        config.insert(String::from("mask"), String::from("0,1,2"));
        config
    }

    #[test]
    fn test_create_minimal() {
        let layer = YoloLayer::from_config(generate_config()).unwrap();
        let layer = layer.as_any().downcast_ref::<YoloLayer>().unwrap();

        assert_eq!(layer.classes, 80);
    }

    #[test]
    #[should_panic(expected = "Key 'classes' is mandatory")]
    fn test_create_fail_classes() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("num"), String::from("2"));
        YoloLayer::from_config(config).unwrap();
    }

    #[test]
    fn test_infer_shape_simple() {
        let shapes: Vec<Rc<dyn Shape>> = vec![Rc::new(LayerShape::from_nchw(32, 3, 10, 20))];
        let mut layer = YoloLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<YoloLayer>().unwrap();

        assert_eq!(*conv_layer.shape().unwrap().dims(), vec![32, 255, 10, 20]);
    }

    #[test]
    #[should_panic(expected = "YoloLayer must have exact one input layer")]
    fn test_infer_shape_invalid_count() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100)),
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100))
            ];

        let mut layer = YoloLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

    #[test]
    #[should_panic(expected = "YoloLayer can be connected only with layer, which produce 4D Tensor with format NCHW")]
    fn test_infer_shape_3d() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nch(32, 3, 128))
            ];

        let mut layer = YoloLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }
    
    #[test]
    fn test_ltype() {
        let layer = YoloLayer::from_config(generate_config()).unwrap();
        assert_eq!(layer.ltype(), LayerType::Yolo);
    }

}