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


//Concat layers across filters or refer previous layer
#[derive(Debug)]
pub struct RouteLayer {
    // Layer name
    name: String,
    // Layer shape
    shape: Option<Rc<dyn Shape>>,
    // Offsets to previous layers
    layers: Vec<i32>
}

const SUPPORTED_FIELDS: [&str; 1] = [
    "layers"
];

impl Layer for RouteLayer {
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
        if input_shapes.len() < 1 {
            return Err(ShapeError{description: String::from("RouteLayer must connect at least 1 layer")})
        }
        let mut new_shape = input_shapes[0].clone();
        let mut iter = input_shapes.iter().skip(1);
        while let Some(shape) = iter.next() {
            new_shape = new_shape.concat(shape.as_ref(), 1)?.into();
        }
    
        self.shape = Some(new_shape);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn from_config(config: HashMap<String, String>) -> Result<Box<dyn Layer>, DeserializationError> {
        let proposed_name = RouteLayer::propose_name();
        let name: String = config.get("name").unwrap_or(&proposed_name).to_string();
        let shape = None;
        let layers = parse_list_field::<i32>(&config, "layers", "RouteLayer")?;
        
        let _ = config.keys().filter(|k| {
            !SUPPORTED_FIELDS.contains(&&k[..])
        }).map(|k| {
            log::warn!("Not supported darknet field during deserialization of '{}'. Field '{}' not recognized", name, k)
        });

        Ok(Box::new(RouteLayer{name, shape, layers}))
    }

}


impl RouteLayer {
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_config() -> HashMap<String, String> {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("layers"), String::from("-4"));
        config
    }

    #[test]
    fn test_create_minimal() {
        let layer = RouteLayer::from_config(generate_config()).unwrap();
        let layer = layer.as_any().downcast_ref::<RouteLayer>().unwrap();
        assert_eq!(layer.layers, [-4]);
    }

    #[test]
    fn test_create_bulk() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("layers"), String::from("-4 ,-5, -6"));
        let layer = RouteLayer::from_config(config).unwrap();
        let layer = layer.as_any().downcast_ref::<RouteLayer>().unwrap();
        assert_eq!(layer.layers, [-4, -5, -6]);
    }

    #[test]
    #[should_panic(expected = "Couldnt parse value '-4.5' in field 'layers' of RouteLayer")]
    fn test_create_fail_parse_float() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("layers"), String::from("-4.5"));
        RouteLayer::from_config(config).unwrap();
    }

    #[test]
    #[should_panic(expected = "Couldnt parse value 'asd' in field 'layers' of RouteLayer")]
    fn test_create_fail_str() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("layers"), String::from("-4,asd,-6"));
        RouteLayer::from_config(config).unwrap();
    }

    #[test]
    #[should_panic(expected = "RouteLayer must connect at least 1 layer")]
    fn test_infer_shape_invalid_count() {
        let shapes: Vec<Rc<dyn Shape>> = vec![];

        let mut layer = RouteLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

    #[test]
    #[should_panic(expected = "Couldnt concat across axis 3")]
    fn test_infer_shape_invalid_shapes() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100)),
            Rc::new(LayerShape::from_nchw(32, 3, 128, 104))
            ];


        let mut layer = RouteLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

    #[test]
    fn test_infer_shape() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100))
            ];

        let mut layer = RouteLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
        assert_eq!(*layer.shape().unwrap().dims(), vec![32, 3, 128, 100]);
    }


    #[test]
    fn test_infer_shape_bulk() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100)),
            Rc::new(LayerShape::from_nchw(32, 7, 128, 100)),
            Rc::new(LayerShape::from_nchw(32, 90, 128, 100))
            ];

        let mut layer = RouteLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
        assert_eq!(*layer.shape().unwrap().dims(), vec![32, 100, 128, 100]);
    }

}