use std::{
    collections::HashMap,
    self,
    any::Any,
    sync::atomic::{Ordering},
    rc::Rc,
    cell::RefCell
};

use crate::nn::shape::*;
use crate::nn::{Layer, LayerType, errors::*, BuildInformation};
use crate::parsers::{DeserializationError, parse_numerical_field, ensure_positive};
use crate::cudnn::{cudnnHandle_t, cudnnDataType, Tensor, DevicePtr};
use crate::nn::ops::{LayerOp, OutputTensor, UpsampleOp};
use crate::nn::{CUDNNEngine, Engine};


//Nearest neighbor upscale
#[derive(Debug)]
pub struct UpsampleLayer {
    // Layer name
    name: String,
    // Layer shape
    shape: Option<Rc<dyn Shape>>,
    // Window stride
    stride: usize,
    // Scale values
    scale: f32
}

const SUPPORTED_FIELDS: [&str; 2] = [
    "stride",
    "scale"
];

impl Layer for UpsampleLayer {
    fn name(&self) -> String {
        self.name.to_string()
    }

    fn shape(&self) -> Option<Rc<dyn Shape>> {
        self.shape.clone()
    }

    fn propose_name() -> String {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!("Upsample_{}", COUNTER.fetch_add(1, Ordering::Relaxed))
    }
 
    fn infer_shape(&mut self, input_shapes: Vec<Rc<dyn Shape>>) -> Result<(), ShapeError> {
        if input_shapes.len() != 1 {
            return Err(ShapeError(String::from("UpsampleLayer must have exact one input layer")))
        }
        let input_shape = &input_shapes[0];
        if input_shape.dims().len() != 4 {
            return Err(ShapeError(String::from("UpsampleLayer can be connected only with layer, which produce 4D Tensor with format NCHW")))
        }

        self.shape = Some(Rc::new(LayerShape::from_nchw(input_shape.N(),
            input_shape.C(),
            input_shape.H().unwrap() * self.stride, 
            input_shape.W().unwrap() * self.stride
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
        let proposed_name = UpsampleLayer::propose_name();
        let name: String = config.get("name").unwrap_or(&proposed_name).to_string();
        let shape = None;
        let stride = parse_numerical_field::<usize>(&config, "stride", true, None)?.unwrap();
        // #TODO: add optimisations when stride == 1
        ensure_positive(stride, "stride", "UpsampleLayer")?;
        let scale = parse_numerical_field::<f32>(&config, "scale", false, Some(1.))?.unwrap();
        
        let _ = config.keys().filter(|k| {
            !SUPPORTED_FIELDS.contains(&&k[..])
        }).map(|k| {
            log::warn!("Not supported darknet field during deserialization of '{}'. Field '{}' not recognized", name, k)
        });

        Ok(Box::new(UpsampleLayer{name, shape, stride, scale}))
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Upsample
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
        let data_type = engine.borrow().dtype();
        let mut operations: Vec<Box<dyn LayerOp>> = Vec::new();

        let shape = self.shape().unwrap();
        let ptr = Rc::new(RefCell::new(
            DevicePtr::new(data_type.clone(), shape.size()).map_err(|e| {
                BuildError::Runtime(e)
            })?
        ));
        let tensor_shape: Box<dyn Shape> = Box::new(LayerShape::new(shape.dims()));
        let tensor = Rc::new(RefCell::new(
            Tensor::new(tensor_shape, ptr).map_err(|e| {
                BuildError::Runtime(e)
            })?
        ));


        operations.push(
            Box::new(UpsampleOp::new(
                engine.borrow().context(),
                info[0].tensor.clone(),
                tensor.clone(),
                self.stride,
                self.scale
            ).map_err(|e| {
                BuildError::Runtime(e)
            })?)
        );
        engine.borrow_mut().add_layer(operations, BuildInformation{tensor, reusable});

        Ok(())
    }

}


impl UpsampleLayer {
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_config() -> HashMap<String, String> {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("stride"), String::from("3"));
        config
    }

    #[test]
    fn test_create_minimal() {
        let layer = UpsampleLayer::from_config(generate_config()).unwrap();
        let layer = layer.as_any().downcast_ref::<UpsampleLayer>().unwrap();

        assert_eq!(layer.stride, 3);
        assert_eq!(layer.scale, 1.);
    }

    #[test]
    fn test_create_sized() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("stride"), String::from("5"));
        config.insert(String::from("scale"), String::from("9."));

        let layer = UpsampleLayer::from_config(config).unwrap();
        let layer = layer.as_any().downcast_ref::<UpsampleLayer>().unwrap();
        assert_eq!(layer.stride, 5);
        assert_eq!(layer.scale, 9.);
    }

    #[test]
    #[should_panic(expected = "Key 'stride' is mandatory")]
    fn test_create_fail_parse_float() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("size"), String::from("2"));
        UpsampleLayer::from_config(config).unwrap();
    }
    #[test]
    fn test_infer_shape_simple() {
        let shapes: Vec<Rc<dyn Shape>> = vec![Rc::new(LayerShape::from_nchw(32, 3, 10, 20))];
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("stride"), String::from("3"));


        let mut layer = UpsampleLayer::from_config(config).unwrap();
        layer.infer_shape(shapes).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<UpsampleLayer>().unwrap();

        assert_eq!(*conv_layer.shape().unwrap().dims(), vec![32, 3, 30, 60]);
    }

    #[test]
    #[should_panic(expected = "UpsampleLayer must have exact one input layer")]
    fn test_infer_shape_invalid_count() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100)),
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100))
            ];

        let mut layer = UpsampleLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

    #[test]
    #[should_panic(expected = "UpsampleLayer can be connected only with layer, which produce 4D Tensor with format NCHW")]
    fn test_infer_shape_3d() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nch(32, 3, 128))
            ];

        let mut layer = UpsampleLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

    #[test]
    fn test_layer_type() {
        let layer = UpsampleLayer::from_config(generate_config()).unwrap();
        assert_eq!(layer.layer_type(), LayerType::Upsample);
    }

}