use std::{
    collections::HashMap,
    self,
    any::Any,
    sync::atomic::{Ordering},
    rc::Rc
};

use crate::nn::shape::*;
use crate::nn::Layer;
use crate::parsers::{DeserializationError, parse_numerical_field, ensure_positive};


//Input layer for most NNs
#[derive(Debug)]
pub struct MaxpoolLayer {
    // Layer name
    name: String,
    // Layer shape
    shape: Option<Rc<dyn Shape>>,
    // Window stride
    stride: usize,
    // Window sizeensure_positive
    size: usize,
    // Padding size
    padding: usize,
}

const SUPPORTED_FIELDS: [&str; 3] = [
    "stride",
    "size",
    "padding",
];

impl Layer for MaxpoolLayer {
    fn name(&self) -> String {
        self.name.to_string()
    }

    fn shape(&self) -> Option<Rc<dyn Shape>> {
        self.shape.clone()
    }

    fn propose_name() -> String {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!("Route_{}", COUNTER.fetch_add(1, Ordering::Relaxed))
    }
 
    fn infer_shape(&mut self, input_shapes: Vec<Rc<dyn Shape>>) -> Result<(), ShapeError> {
        if input_shapes.len() != 1 {
            return Err(ShapeError{description: String::from("MaxpoolLayer must have exact one input layer")})
        }
        let input_shape = &input_shapes[0];
        if input_shape.dims().len() != 4 {
            return Err(ShapeError{description: String::from("MaxpoolLayer can be connected only with layer, which produce 4D Tensor with format NCHW")})
        }
        
        let h = (input_shape.H().unwrap() + self.padding - self.size) / self.stride + 1;
        if h < 1 {
            return Err(ShapeError{description: String::from("Resulting height is less then 1")});
        }

        let w = (input_shape.W().unwrap() + self.padding - self.size) / self.stride + 1;
        if w < 1 {
            return Err(ShapeError{description: String::from("Resulting width is less then 1")});
        }

        self.shape = Some(Rc::new(LayerShape::from_nchw(input_shape.N(), input_shape.C(), h, w)));
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn from_config(config: HashMap<String, String>) -> Result<Box<dyn Layer>, DeserializationError> {
        let proposed_name = MaxpoolLayer::propose_name();
        let name: String = config.get("name").unwrap_or(&proposed_name).to_string();
        let shape = None;
        let stride = parse_numerical_field::<usize>(&config, "stride", true, None)?.unwrap();
        ensure_positive(stride, "stride", "MaxpoolLayer")?;
        let size = parse_numerical_field::<usize>(&config, "size", false, Some(stride))?.unwrap();
        ensure_positive(stride, "size", "MaxpoolLayer")?;
        let padding = parse_numerical_field::<usize>(&config, "padding", false, Some(size - 1))?.unwrap();
        
        let _ = config.keys().filter(|k| {
            !SUPPORTED_FIELDS.contains(&&k[..])
        }).map(|k| {
            log::warn!("Not supported darknet field during deserialization of '{}'. Field '{}' not recognized", name, k)
        });

        Ok(Box::new(MaxpoolLayer{name, shape, stride, size, padding}))
    }

}


impl MaxpoolLayer {
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
        let layer = MaxpoolLayer::from_config(generate_config()).unwrap();
        let layer = layer.as_any().downcast_ref::<MaxpoolLayer>().unwrap();

        assert_eq!(layer.stride, 3);
        assert_eq!(layer.size, 3);
    }

    #[test]
    fn test_create_sized() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("stride"), String::from("5"));
        config.insert(String::from("size"), String::from("9"));

        let layer = MaxpoolLayer::from_config(config).unwrap();
        let layer = layer.as_any().downcast_ref::<MaxpoolLayer>().unwrap();
        assert_eq!(layer.stride, 5);
        assert_eq!(layer.size, 9);
    }

    #[test]
    #[should_panic(expected = "Key 'stride' is mandatory")]
    fn test_create_fail_parse_float() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("size"), String::from("2"));
        MaxpoolLayer::from_config(config).unwrap();
    }
    #[test]
    fn test_infer_shape_simple() {
        let shapes: Vec<Rc<dyn Shape>> = vec![Rc::new(LayerShape::from_nchw(32, 3, 128, 128))];
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("stride"), String::from("2"));


        let mut layer = MaxpoolLayer::from_config(config).unwrap();
        layer.infer_shape(shapes).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<MaxpoolLayer>().unwrap();

        assert_eq!(*conv_layer.shape().unwrap().dims(), vec![32, 3, 64, 64]);
    }

    #[test]
    fn test_infer_shape_sized() {
        let mut config = generate_config();
        config.insert(String::from("size"), String::from("8"));
        config.insert(String::from("stride"), String::from("4"));
        let shapes: Vec<Rc<dyn Shape>> = vec![Rc::new(LayerShape::from_nchw(32, 3, 128, 100))];


        let mut layer = MaxpoolLayer::from_config(config).unwrap();
        layer.infer_shape(shapes).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<MaxpoolLayer>().unwrap();

        assert_eq!(*conv_layer.shape().unwrap().dims(), vec![32, 3, 32, 25]);
    }
    #[test]
    #[should_panic(expected = "MaxpoolLayer must have exact one input layer")]
    fn test_infer_shape_invalid_count() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100)),
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100))
            ];

        let mut layer = MaxpoolLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

    #[test]
    #[should_panic(expected = "MaxpoolLayer can be connected only with layer, which produce 4D Tensor with format NCHW")]
    fn test_infer_shape_3d() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nch(32, 3, 128))
            ];

        let mut layer = MaxpoolLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

}