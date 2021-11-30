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
use crate::nn::{Layer, LayerType, errors::*, BuildInformation};
use crate::parsers::{DeserializationError, parse_numerical_field, ensure_positive, parse_list_field};
use crate::cudnn::{cudnnHandle_t, cudnnDataType, Tensor, DevicePtr};
use crate::nn::ops::{LayerOp, OutputTensor, ConvertOp};

//Yolo head layer
#[derive(Debug)]
pub struct YoloLayer {
    // Layer name
    name: String,
    // Layer shape
    shape: Option<Rc<dyn Shape>>,
    // Window stride
    classes: usize,
    // List of operations
    operations: Vec<Box<dyn LayerOp>>,
    // Can be reusable
    reusable: bool,
    // Output tensor
    tensor: Option<OutputTensor>,
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

        let tensor = None;
        let operations = vec![];
        let reusable = false;

        Ok(Box::new(YoloLayer{name, shape,
            classes, anchors,
            tensor, operations,
            reusable, scale
        }))
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Yolo
    }

    fn get_build_information(&self) -> BuildInformation {
        BuildInformation{tensor: self.tensor.as_ref().unwrap().clone(), reusable: self.reusable}
    }

    fn build(&mut self, 
        context: Rc<cudnnHandle_t>,
        data_type: &cudnnDataType,
        info: Vec<BuildInformation>,
        _has_depend_layers: bool
    ) -> Result<(), BuildError> {
        if data_type == &cudnnDataType::FLOAT {
            self.tensor = Some(info[0].tensor.clone());
        } else {
            let shape = self.shape().unwrap();
            let ptr = Rc::new(RefCell::new(
                DevicePtr::new(cudnnDataType::FLOAT, shape.size()).map_err(|e| {
                    BuildError::Runtime(e)
                })?
            ));

            let tensor_shape: Box<dyn Shape> = Box::new(LayerShape::new(shape.dims()));
            let tensor = Rc::new(RefCell::new(
                Tensor::new(tensor_shape, ptr).map_err(|e| {
                    BuildError::Runtime(e)
                })?
            ));

            self.tensor = Some(tensor.clone());
            self.operations.push(
                Box::new(ConvertOp::new(
                    context.clone(),
                    info[0].tensor.clone(),
                    tensor.clone(),
                ).map_err(|e| {
                    BuildError::Runtime(e)
                })?)
            );
        }
        Ok(())
    }

}

impl YoloLayer {
    
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
    fn test_layer_type() {
        let layer = YoloLayer::from_config(generate_config()).unwrap();
        assert_eq!(layer.layer_type(), LayerType::Yolo);
    }

}