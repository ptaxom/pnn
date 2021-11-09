use std::{
    collections::HashMap,
    self,
    any::Any,
    sync::atomic::{Ordering},
    rc::Rc
};

use crate::nn::shape::*;
use crate::nn::Layer;
use crate::parsers::{DeserializationError, parse_numerical_field};


//Input layer for most NNs
#[derive(Debug)]
pub struct InputLayer {
    // Layer name
    name: String,
    // Layer shape
    shape: Option<Rc<dyn Shape>>,
    // batchless shape dims. After defining batchsize it should become shape
    dims: Vec<usize>
}


impl Layer for InputLayer {
    fn name(&self) -> String {
        self.name.to_string()
    }

    fn shape(&self) -> Option<Rc<dyn Shape>> {
        self.shape.clone()
    }

    fn propose_name() -> String {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!("Input_{}", COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn infer_shape(&mut self, _input_shapes: Vec<Rc<dyn Shape>>) -> Result<(), ShapeError>  {
        Ok(())
    }

    fn from_config(config: HashMap<String, String>) -> Result<Box<dyn Layer>, DeserializationError> {
        let proposed_name = InputLayer::propose_name();
        let name: String = config.get("name").unwrap_or(&proposed_name).to_string();
        let shape = None;
        let channels = parse_numerical_field::<usize>(&config, "channels", true, None)?.unwrap();
        let mut dims = vec![channels];
        if let Some(height) = parse_numerical_field::<usize>(&config, "height", false, None)? {
            dims.push(height);
        }
        if let Some(width) = parse_numerical_field::<usize>(&config, "width", false, None)? {
            match dims.len() {
                2 => dims.push(width),
                _ => return Err(DeserializationError{description: String::from("Couldnt use key 'width' without key 'height' in InputLayer")})
            }
        }
        Ok(Box::new(InputLayer{name, shape, dims}))
    }

}

impl InputLayer {
    // Infer shape by known batchsize
    pub fn set_batchsize(&mut self, batchsize: usize) {
        let mut new_dims = self.dims.clone();
        new_dims.insert(0, batchsize);
        self.shape = Some(Rc::new(LayerShape::new(new_dims)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn generate_config() -> HashMap<String, String> {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("channels"), String::from("3"));
        config
    }

    #[test]
    fn test_deserialize() {
        let layer = InputLayer::from_config(generate_config()).unwrap();
        match layer.shape() {
            None => assert!(true),
            _ => assert!(false)
        }
        // assert_eq!(layer.name(), "Input_0"); // Static variable during tests is not determenistic, but its work, I swear
        let inp_layer = layer.as_any().downcast_ref::<InputLayer>().unwrap();
        assert_eq!(inp_layer.dims, vec![3]);
    }

    #[test]
    fn test_deserialize_names() {        
        let _layer = InputLayer::from_config(generate_config()).unwrap();
        let _layer2 = InputLayer::from_config(generate_config()).unwrap();

        let mut named_config = generate_config();
        named_config.insert(String::from("name"), String::from("CustomName"));
        let layer_named = InputLayer::from_config(named_config).unwrap();

        // assert_eq!(layer.name(), "Input_0");
        // assert_eq!(layer2.name(), "Input_1");
        assert_eq!(layer_named.name(), "CustomName");
    }

    #[test]
    fn test_deserialize_with_height() {
        let mut config = generate_config();
        config.insert(String::from("height"), String::from("5"));
        let layer = InputLayer::from_config(config).unwrap();

        let inp_layer = layer.as_any().downcast_ref::<InputLayer>().unwrap();
        assert_eq!(inp_layer.dims, vec![3, 5]);
    }

    #[test]
    fn test_deserialize_with_height_and_width() {
        let mut config = generate_config();
        config.insert(String::from("height"), String::from("5"));
        config.insert(String::from("width"), String::from("6"));
        let layer = InputLayer::from_config(config).unwrap();

        let inp_layer = layer.as_any().downcast_ref::<InputLayer>().unwrap();
        assert_eq!(inp_layer.dims, vec![3, 5, 6]);
    }

    #[test]
    #[should_panic(expected = "Couldnt use key 'width' without key 'height' in InputLayer")]
    fn test_deserialize_with_width() {
        let mut config = generate_config();
        config.insert(String::from("width"), String::from("6"));
        InputLayer::from_config(config).unwrap();
    }

    #[test]
    #[should_panic(expected = "Couldnt parse '3.5' for key 'height'")]
    fn test_deserialize_wrong_type() {
        let mut config = generate_config();
        config.insert(String::from("height"), String::from("3.5"));
        InputLayer::from_config(config).unwrap();
    }

    #[test]
    #[should_panic(expected = "Key 'channels' is mandatory")]
    fn test_deserialize_empty() {
        let config: HashMap<String, String> = HashMap::new();
        InputLayer::from_config(config).unwrap();
    }

    #[test]
    fn test_batchsize() {
        let mut config = generate_config();
        config.insert(String::from("height"), String::from("1080"));
        config.insert(String::from("width"), String::from("1920"));
        let mut layer = InputLayer::from_config(config).unwrap();

        let inp_layer = layer.as_any_mut().downcast_mut::<InputLayer>().unwrap();
        (*inp_layer).set_batchsize(32);
        let target: Vec<usize> = vec![32, 3, 1080, 1920];
        assert_eq!(*inp_layer.shape().unwrap().dims(), target);
    }
}