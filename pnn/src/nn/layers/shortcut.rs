use std::{
    collections::HashMap,
    self,
    any::Any,
    sync::atomic::{Ordering},
    rc::Rc
};

use crate::nn::shape::*;
use crate::nn::Layer;
use crate::parsers::{DeserializationError, parse_list_field};


//Input layer for most NNs
#[derive(Debug)]
pub struct ShortcutLayer {
    // Layer name
    name: String,
    // Layer shape
    shape: Option<Rc<dyn Shape>>,
    // Offsets to previous layers
    from: Vec<i32>,
    // Activation function
    activation: String
}

const SUPPORTED_FIELDS: [&str; 2] = [
    "from",
    "activation"
];

impl Layer for ShortcutLayer {
    fn name(&self) -> String {
        self.name.to_string()
    }

    fn shape(&self) -> Option<Rc<dyn Shape>> {
        self.shape.clone()
    }

    fn propose_name() -> String {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!("Shortcut_{}", COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    fn infer_shape(&mut self, input_shapes: Vec<Rc<dyn Shape>>) -> Result<(), ShapeError> {
        if input_shapes.len() < 2 {
            return Err(ShapeError{description: String::from("ShortcutLayer must connect at least 2 layers")})
        }
        let input_dim = input_shapes[0].dims();
        for shape in &input_shapes {
            if shape.dims().iter().cmp(input_dim.iter()) != std::cmp::Ordering::Equal {
                return Err(ShapeError{description: String::from("All input tensors at ShortcutLayer must have same shape")});
            }
        }
        self.shape = Some(Rc::new(LayerShape::new(input_dim.clone())));
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn from_config(config: HashMap<String, String>) -> Result<Box<dyn Layer>, DeserializationError> {
        let proposed_name = ShortcutLayer::propose_name();
        let name: String = config.get("name").unwrap_or(&proposed_name).to_string();
        let shape = None;
        let from = parse_list_field::<i32>(&config, "from", "ShortcutLayer")?;

        let activation: String = config.get("activation").unwrap_or(&String::from("linear")).to_string();
        
        let _ = config.keys().filter(|k| {
            !SUPPORTED_FIELDS.contains(&&k[..])
        }).map(|k| {
            log::warn!("Not supported darknet field during deserialization of '{}'. Field '{}' not recognized", name, k)
        });

        Ok(Box::new(ShortcutLayer{name, shape, from, activation}))
    }

}


impl ShortcutLayer {
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_config() -> HashMap<String, String> {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("from"), String::from("-4"));
        config
    }

    #[test]
    fn test_create_minimal() {
        let layer = ShortcutLayer::from_config(generate_config()).unwrap();
        let layer = layer.as_any().downcast_ref::<ShortcutLayer>().unwrap();
        assert_eq!(layer.from, [-4]);
        assert_eq!(layer.activation, "linear");
    }

    #[test]
    fn test_create_bulk() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("from"), String::from("-4,-5,-6"));
        let layer = ShortcutLayer::from_config(config).unwrap();
        let layer = layer.as_any().downcast_ref::<ShortcutLayer>().unwrap();
        assert_eq!(layer.from, [-4, -5, -6]);
        assert_eq!(layer.activation, "linear");
    }

    #[test]
    #[should_panic(expected = "Couldnt parse value '-4.5' in field 'from' of ShortcutLayer")]
    fn test_create_fail_parse_float() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("from"), String::from("-4.5"));
        ShortcutLayer::from_config(config).unwrap();
    }

    #[test]
    #[should_panic(expected = "Couldnt parse value 'asd' in field 'from' of ShortcutLayer")]
    fn test_create_fail_str() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("from"), String::from("-4,asd,-6"));
        ShortcutLayer::from_config(config).unwrap();
    }

    #[test]
    #[should_panic(expected = "ShortcutLayer must connect at least 2 layers")]
    fn test_infer_shape_invalid_count() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100))
            ];


        let mut layer = ShortcutLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

    #[test]
    #[should_panic(expected = "All input tensors at ShortcutLayer must have same shape")]
    fn test_infer_shape_invalid_shapes() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100)),
            Rc::new(LayerShape::from_nchw(32, 3, 128, 101))
            ];


        let mut layer = ShortcutLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

    #[test]
    fn test_infer_shape() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100)),
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100))
            ];


        let mut layer = ShortcutLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
        assert_eq!(*layer.shape().unwrap().dims(), vec![32, 3, 128, 100]);
    }

}