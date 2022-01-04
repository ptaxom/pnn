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
use crate::parsers::{DeserializationError, parse_list_field};
use crate::cudnn::{Tensor, DevicePtr};
use crate::nn::{CUDNNEngine, TRTBuilder};
use crate::nn::ops::{LayerOp, InputTensor, RouteOp};


//Concat layers across filters or refer previous layer
#[derive(Debug)]
pub struct RouteLayer {
    // Layer name
    name: String,
    // Layer shape
    shape: Option<Rc<dyn Shape>>,
    // Offsets to previous layers
    layers: Vec<i32>,
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
            return Err(ShapeError(String::from("RouteLayer must connect at least 1 layer")))
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

    fn ltype(&self) -> LayerType {
        LayerType::Route
    }

    fn input_indices(&self, position: usize) -> Result<Vec<usize>, BuildError> {
        if position == 0 {
            return Err(BuildError::Deserialization(DeserializationError(String::from("Couldnt compute input index for first layer"))))
        }
        let indeces: Result<Vec<usize>, BuildError> = self.layers.iter().map(|x| {
            // -1 to compensate input layer during absolute index. # TODO: fix it
            let index: i32 = if *x > 0i32 {*x + 1} else {position as i32 + *x};
            if index >= position as i32 || index < 0 {
                return Err(BuildError::Deserialization(DeserializationError(format!("Couldnt reffer to {} from '{}'", index, self.name))))
            }
            Ok(index as usize)
        }).collect();
        indeces
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
            Box::new(RouteOp::new(
                engine.borrow().context(),
                info.iter().map(|i| {
                    i.tensor.clone()
                }).collect::<Vec<InputTensor>>(),
                tensor.clone()
            ).map_err(|e| {
                BuildError::Runtime(e)
            })?)
        );
        engine.borrow_mut().add_layer(operations, BuildInformation{tensor, reusable});
        Ok(())
    }

    fn build_trt(&mut self, 
        engine: Rc<RefCell<TRTBuilder>>,
        indeces: Vec<usize>
    ) -> Result<(), BuildError> {
        let mut engine = engine.borrow_mut();
        let indeces: Vec<usize> = indeces.iter().map(|x| {
            engine.last_op_id(*x)
        }).collect();
        let id = engine.add_route(&indeces)?;
        engine.finilize_layer(id);
        Ok(())
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

    #[test]
    fn test_ltype() {
        let layer = RouteLayer::from_config(generate_config()).unwrap();
        assert_eq!(layer.ltype(), LayerType::Route);
    }

    #[test]
    fn test_input_indeces() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("layers"), String::from("-4 ,-5, -6"));
        let layer = RouteLayer::from_config(config).unwrap();
        assert_eq!(layer.input_indices(20).unwrap(), [16usize, 15usize, 14usize]);
    }

    #[test]
    fn test_input_indeces_absolute() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("layers"), String::from("-4 ,-5, -6, 2"));
        let layer = RouteLayer::from_config(config).unwrap();
        assert_eq!(layer.input_indices(20).unwrap(), [16usize, 15usize, 14usize, 3usize]);
    }

    #[test]
    #[should_panic]
    fn test_input_indeces_fail() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("layers"), String::from("-4 ,-5, -6"));
        let layer = RouteLayer::from_config(config).unwrap();
        layer.input_indices(5).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_input_indeces_fail_forward_ref() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("layers"), String::from("-4 ,-5, -6, 22"));
        let layer = RouteLayer::from_config(config).unwrap();
        layer.input_indices(20).unwrap();
    }


}